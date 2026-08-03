#include "segmentation.hpp"

lane_seg::Segmenter::Segmenter(std::shared_ptr<TwinLite> model,
    std::string ipAddress, std::string svoPath,
    DebugMode debugMode) :
    model(model), debugMode(debugMode), ipAddress(ipAddress), optSvoPath(svoPath), valid(false) {

    if (!model->isLoaded()) {
        globalLogger.error("Failed to properly load model!");
        return;
    }

    if (!setupZED()) {
        globalLogger.error("Failed to set up ZED camera!");
        return;
    }

    valid = true;
}

bool lane_seg::Segmenter::runLoop() {
    if (!valid) {
        globalLogger.error("Failed to start detector from invalid state");
        return false;
    }

    resolution = zed->getRetrieveImageResolution();    // can technically change after disconnect / reconnect but if it does we have bigger problems
    auto runtimeParams = sl::RuntimeParameters();

    sl::Mat leftBGR(resolution, sl::MAT_TYPE::U8_C4, (debugMode == DebugMode::NONE ? sl::MEM::GPU : sl::MEM::BOTH));
    sl::Mat depthXYZ(resolution, sl::MAT_TYPE::F32_C4, (debugMode == DebugMode::NONE ? sl::MEM::GPU : sl::MEM::BOTH));

    cv::Mat cpuFrame;
    if (debugMode != DebugMode::NONE) {
        cpuFrame = cv::Mat(
            leftBGR.getHeight(), leftBGR.getWidth(), CV_8UC4, 
            leftBGR.getPtr<sl::uchar4>(sl::MEM::CPU), 
            leftBGR.getStepBytes(sl::MEM::CPU)
        );
    }

    cv::VideoWriter videoOut;
    if (debugMode == DebugMode::VISUALIZE) {
        cv::namedWindow("Object Detection", cv::WINDOW_NORMAL);
        globalLogger.log(Severity::kINFO, "Press \"q\" to exit the video display");
    } else if (debugMode == DebugMode::VIDEO_OUT) {
        const std::string filename = "objdet_output" + std::to_string(std::time(nullptr)) + ".avi";

        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        videoOut.open(filename, fourcc, 30.0, cv::Size(resolution.width, resolution.height), true);
        if (!videoOut.isOpened()) {
            globalLogger.error("Debug mode set to VIDEO_OUT, but failed to open video output file for writing...");
            return false;
        }
    }

    cudaStream_t frameStream, copyStream;
    cudaStreamCreate(&frameStream);
    cudaStreamCreate(&copyStream);

    std::size_t scoresSize = (std::size_t)model->getConfig().outputSize.area() * 2;
    std::vector<float> llScores, daScores;
    llScores.resize(scoresSize);
    daScores.resize(scoresSize);

    std::vector<LaneSegment> toSerialize;
    std::string json;

    std::uint64_t debugFrameCount = 0;

    while (true) {
        Timer timer("Frame Time");
        
        auto errorCode = zed->grab(runtimeParams); 
        if (errorCode != sl::ERROR_CODE::SUCCESS) {
            if (errorCode == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
                globalLogger.log(Severity::kINFO, "End of SVO file reached, exiting...");
                break;
            }
            globalLogger.log(Severity::kWARNING, "ZED camera frame dropped with below error, attempting reconnect:");
			globalLogger.log(Severity::kWARNING, sl::toString(errorCode));

            // ZED camera reconnect fix
            if (!setupZED()) {
				globalLogger.error("All ZED camera reconnect attempts failed, exiting...");
                break;
            }

            globalLogger.log(Severity::kINFO, "ZED camera reconnected successfully, continuing...");
            continue;
        }
        
        errorCode = zed->retrieveImage(leftBGR, sl::VIEW::LEFT_BGR, (debugMode == DebugMode::NONE ? sl::MEM::GPU : sl::MEM::BOTH), resolution, frameStream);
        if (errorCode != sl::ERROR_CODE::SUCCESS) {
            globalLogger.error("Failed to retreive ZED camera frame with error code " + std::to_string((int)errorCode) + ": " + std::string(sl::toString(errorCode).c_str()));
            continue;
        }
        errorCode = zed->retrieveMeasure(depthXYZ, sl::MEASURE::XYZ, (debugMode == DebugMode::NONE ? sl::MEM::GPU : sl::MEM::BOTH), resolution, frameStream);
        if (errorCode != sl::ERROR_CODE::SUCCESS) {
            globalLogger.error("Failed to retreive ZED depth map with error code " + std::to_string((int)errorCode) + ": " + std::string(sl::toString(errorCode).c_str()));
            continue;
        }

        if (!model->predict(leftBGR, depthXYZ, llScores, daScores, debugMode != DebugMode::NONE)) {
            globalLogger.error("Failed to queue model prediction");
            break;
        }

        serializeSegments(toSerialize, json);
        std::printf("OBJ_DATA: %s\n\n", json.c_str());              // write to stdout like this, supposedly faster
        globalLogger.log(Severity::kINFO, "ZED FPS: " + std::to_string(zed->getCurrentFPS()));

        if (debugMode != DebugMode::NONE) {
            cv::Mat resized;
            cv::resize(cpuFrame, resized, model->getConfig().inputSize);
			
            constexpr float fontScale = 0.75f;
			const auto baseColor = cv::Scalar(0,0,0), frontColor = cv::Scalar(180, 180, 180);
			constexpr std::uint32_t baseWeight = 3, frontWeight = 1;
            
			const int height = cv::getTextSize("test", cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).height + 4;
            
            const std::string dynamicInfo[] = {
                "Frame: " + std::to_string(debugFrameCount++),
                "Segments: " + std::to_string(toSerialize.size()),
                "Last frame time: " + std::to_string(Timer::lastTimeUS) + "us",
			};
            
            static const std::string staticInfo[] = { 
                "[[ DEBUG VIEW ]]",
                "Model(s): \"" + model->getFilePath().generic_string() + "\"",
				"Res: (" + std::to_string(resolution.width) + ", " 
                    + std::to_string(resolution.height) + ") => (" 
                    + std::to_string(model->getConfig().inputSize.width) + ", " 
                    + std::to_string(model->getConfig().inputSize.height) + ")",
				"Source: " + std::string(zed->getSVOPosition() > 0 ? "SVO file" : "live camera"),
                "Debug mode: " + std::string(debugMode == DebugMode::VISUALIZE ? "VISUALIZE" : "VIDEO_OUT") + "(" + std::to_string((int)debugMode) + ")",
            };
            
            // poor design but it's fine for debug
            static const int widths[] = {
                std::abs(cv::getTextSize(staticInfo[0], cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).width),
                std::abs(cv::getTextSize(staticInfo[1], cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).width),
                std::abs(cv::getTextSize(staticInfo[2], cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).width),
                std::abs(cv::getTextSize(staticInfo[3], cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).width),
                std::abs(cv::getTextSize(staticInfo[4], cv::FONT_HERSHEY_DUPLEX, fontScale, baseWeight, nullptr).width)
            };
            
			static_assert(std::size(staticInfo) == std::size(widths), "VISUALIZE debug output static info and widths array size mismatch");
            
			cv::putText(cpuFrame, staticInfo[0], cv::Point((cpuFrame.cols - widths[0]) / 2, 60), cv::FONT_HERSHEY_DUPLEX, fontScale, baseColor, baseWeight);
            cv::putText(cpuFrame, staticInfo[0], cv::Point((cpuFrame.cols - widths[0]) / 2, 60), cv::FONT_HERSHEY_DUPLEX, fontScale, frontColor, frontWeight);
            
            for (std::size_t index = 1; index < std::size(staticInfo); index++) {
                auto point = cv::Point(cpuFrame.cols - widths[index] - 50, 80 + ((index - 1) * height));
                
                cv::putText(cpuFrame, staticInfo[index], point, cv::FONT_HERSHEY_DUPLEX, fontScale, baseColor, baseWeight);
                cv::putText(cpuFrame, staticInfo[index], point, cv::FONT_HERSHEY_DUPLEX, fontScale, frontColor, frontWeight);
            }
            
            for (std::size_t index = 0; index < std::size(dynamicInfo); index++) {
                auto point = cv::Point(50, 80 + (index * height));
            
                cv::putText(cpuFrame, dynamicInfo[index], point, cv::FONT_HERSHEY_DUPLEX, fontScale, baseColor, baseWeight);
                cv::putText(cpuFrame, dynamicInfo[index], point, cv::FONT_HERSHEY_DUPLEX, fontScale, frontColor, frontWeight);
            }
            
            if (debugMode == DebugMode::VISUALIZE) {
                cv::imshow("Lane Segmentation (build " + std::string(__DATE__) + ")", cpuFrame);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 27) {
                    globalLogger.log(Severity::kINFO, "Escape key pressed...");
                    break;
                }
            } else if (debugMode == DebugMode::VIDEO_OUT) {
                videoOut.write(cpuFrame);
                std::string countString = std::to_string(debugFrameCount);
				countString.insert(countString.begin(), 6 - countString.length(), '0'); // pad with zeros
				//cv::imwrite("frame_" + countString + ".png", cpuFrame);
            }

            cv::Mat netOverlay = resized.clone();
            cv::Mat geoOverlay = resized.clone();

            // Argmax for Net (Simplified C++ equivalent mapping CHW flat arrays back to HWC space)
            // TODO: make this a visualization kernel
            int plane_size = model->getConfig().inputSize.area();
            int outputW = model->getConfig().outputSize.width;
            int outputH = model->getConfig().outputSize.height;

            const cv::Mat& groundMask = model->getGroundMask();
            const cv::Mat& obstacleMask = model->getObstacleMask();

            for (int i = 0; i < plane_size; ++i) {
                if (daScores[1 * plane_size + i] > daScores[0 * plane_size + i]) {
                    netOverlay.at<cv::Vec3b>(i / outputW, i % outputW) = cv::Vec3b(255, 0, 0); // Blue (Drivable)
                }
                if (llScores[1 * plane_size + i] > llScores[0 * plane_size + i]) {
                    netOverlay.at<cv::Vec3b>(i / outputW, i % outputW) = cv::Vec3b(0, 255, 0); // Green (Lines)
                }

                // Geo Overlay mapping
                int r = i / outputW, c = i % outputW;
                if (groundMask.at<uchar>(r, c)) geoOverlay.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 255); // Yellow
                if (obstacleMask.at<uchar>(r, c)) geoOverlay.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255); // Red
            }

            cv::addWeighted(resized, 0.5, netOverlay, 0.5, 0, netOverlay);
            cv::addWeighted(resized, 0.5, geoOverlay, 0.5, 0, geoOverlay);

            cv::imshow("TwinLiteNet+ TensorRT", netOverlay);
            cv::imshow("ZED Depth Geometry", geoOverlay);

        }
    }

    if (videoOut.isOpened()) {
        videoOut.release();
    }

    cudaStreamSynchronize(frameStream);
    cudaStreamSynchronize(copyStream);

    zed->close();
    cv::destroyAllWindows();
    cudaStreamDestroy(frameStream);
    cudaStreamDestroy(copyStream);

    leftBGR.free();
    depthXYZ.free();

    globalLogger.log(Severity::kINFO, "Processing complete");
    return true;
}

bool lane_seg::Segmenter::setupZED(std::size_t maxRetries, double retryDelay) {
    sl::InitParameters init;

    init.camera_resolution = sl::RESOLUTION::HD1080;
    init.depth_mode = sl::DEPTH_MODE::NEURAL;
    init.coordinate_units = sl::UNIT::METER;
    init.camera_fps = 30.0f;                            // "switch to 60 for summer comp"
    init.depth_minimum_distance = 0.5f;                 // meters
    init.depth_maximum_distance = 40.0f;                // ^^^^^^
    init.depth_stabilization = 0;
    init.async_grab_camera_recovery = true;
    init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

	//init.svo_real_time_mode = false; // process frames as fast as possible

    switch (optSvoPath.empty()) {
        case false: {
            init.input.setFromSVOFile(optSvoPath.c_str());
            globalLogger.log(Severity::kINFO, "ZED camera using SVO file input from path: \"" + optSvoPath + "\"");
            break;
        }
        case true: {
            
            // can't you just regex
            if (!ipAddress.empty()) {
                // Verify address contains only digits, dots and optional colon/port

                std::string stripped = ipAddress;
                stripped.erase(std::remove_if(stripped.begin(), stripped.end(),
                    [](char c) {
                    return c == '.' || c == ':';
                }), stripped.end());
                bool valid = (!stripped.empty() && std::all_of(stripped.begin(), stripped.end(), isdigit))      // Stripped string must be only numeric
                    && (std::count(ipAddress.cbegin(), ipAddress.cend(), '.') == 3)                             // Exactly three dots
                    && (std::count(ipAddress.cbegin(), ipAddress.cend(), ':') <= 1);                            // At most one colon

                if (valid) {
                    auto colonPos = ipAddress.find(':');
                    if (colonPos != std::string::npos) {
                        std::string ip = ipAddress.substr(0, colonPos);
                        int port = std::stoi(ipAddress.substr(colonPos + 1));
                        init.input.setFromStream(ip.c_str(), port);
                        globalLogger.log(Severity::kINFO, "Using stream input, IP: \"" + ip + "\"");
                    } else {
                        init.input.setFromStream(ipAddress.c_str());
                        globalLogger.log(Severity::kINFO, "Using stream input, IP: \"" + ipAddress + "\"");
                    }
                } else {
                    globalLogger.error("Invalid IP address for ZED camera setup, using live stream...");
                }
            }
            break;
        }
    }

    zed = std::make_unique<sl::Camera>();
    for (std::size_t retryCount = 0; retryCount < maxRetries; retryCount++) {
		globalLogger.log(Severity::kVERBOSE, "Attempting to open ZED camera (attempt " + std::to_string(retryCount + 1) + ")...");
        if (zed->open(init) == sl::ERROR_CODE::SUCCESS) {
            zed->setCameraSettings(sl::VIDEO_SETTINGS::LED_STATUS, 1);
            globalLogger.log(Severity::kINFO, "ZED camera successfully opened!");
            return true;
        }

        globalLogger.log(Severity::kWARNING, "failed to open ZED camera, retrying in " + std::to_string(retryDelay) + " seconds... (attempt " + std::to_string(retryCount + 1) + ")");
        std::this_thread::sleep_for(std::chrono::duration<double>(retryDelay));
    }

    globalLogger.error("Repeatedly unable to establish connection with ZED camera after " + std::to_string(maxRetries) + " total attempts");
    zed->close();
    return false;
}

void lane_seg::Segmenter::serializeSegments(const std::vector<LaneSegment>& tracks, std::string& json) {
    json = "{ UNIMPLEMENTED }";
    
    //if (tracks.empty()) {
    //    static const std::string none = "\"0\": {\"obj_cartesian_coords\": [" + DEF_DICT_STR + ", " + DEF_DICT_STR + ", " + DEF_DICT_STR + "], " +
    //        "\"obj_class\": 255, \"obj_type\": 255, " +
    //        "\"obj_width\": " + DEF_DICT_STR + ", " +
    //        "\"obj_height\": " + DEF_DICT_STR + "}";
    //    json = none;
    //    return;
    //}
    //
    //json.clear();
    //json.reserve(tracks.size() * 200); // rough estimate
    //
    //// Helper lambda for appending numbers
    //std::array<char, 32> numBuffer{};
    //auto append_num = [&](auto value) {
    //    if (auto [ptr, ec] = std::to_chars(numBuffer.data(), numBuffer.data() + numBuffer.size(), value); ec == std::errc()) {
    //        json.append(numBuffer.data(), ptr - numBuffer.data());
    //    }
    //};
    //
    //using namespace json_literals;
    //
    //json.append(S_BRACE_O);
    //for (std::size_t index = 0; index < tracks.size(); index++) {
    //    const TrackedObject& track = tracks[index];
    //
    //    // indexed key
    //    json.append(S_QUOTE);
    //    append_num(index);
    //    json.append(S_QUOTE);
    //    json.append(S_COLON);
    //
    //    // default / invalid
    //    if (tracks[index].coords[0] == DEF_DICT_VAL) {
    //        json.append(S_COORDS);
    //        append_num(DEF_DICT_VAL); json.append(S_COMMA);
    //        append_num(DEF_DICT_VAL); json.append(S_COMMA);
    //        append_num(DEF_DICT_VAL); json.append(S_BRACKET_C);
    //
    //        json.append(S_CLASS); json.append(S_255);
    //        json.append(S_TYPE); json.append(S_255);
    //        json.append(S_WIDTH); append_num(DEF_DICT_VAL);
    //        json.append(S_HEIGHT); append_num(DEF_DICT_VAL);
    //    } else {
    //        json.append(S_COORDS);
    //        append_num(track.coords[0]); json.append(S_COMMA);
    //        append_num(track.coords[1]); json.append(S_COMMA);
    //        append_num(track.coords[2]); json.append(S_BRACKET_C);
    //
    //        json.append(S_CLASS); append_num((int)track.classID);
    //        json.append(S_TYPE); append_num((int)track.customLabel);
    //
    //        if (tracks[index].classID == ObjectType::TRAFFIC_LIGHT || tracks[index].customLabel == CustomLabel::NONE) {
    //            const auto& dims = OBJ_DIMS_CLASS.find(tracks[index].classID);
    //
    //            json.append(S_WIDTH);
    //            append_num(dims != OBJ_DIMS_CLASS.end() ? dims->second[0] : DEF_DICT_VAL);
    //            json.append(S_HEIGHT);
    //            append_num(dims != OBJ_DIMS_CLASS.end() ? dims->second[1] : DEF_DICT_VAL);
    //        } else {
    //            const auto& dims = OBJ_DIMS_TYPE.find(tracks[index].customLabel);
    //
    //            json.append(S_WIDTH);
    //            append_num(dims != OBJ_DIMS_TYPE.end() ? dims->second[0] : DEF_DICT_VAL);
    //            json.append(S_HEIGHT);
    //            append_num(dims != OBJ_DIMS_TYPE.end() ? dims->second[1] : DEF_DICT_VAL);
    //        }
    //    }
    //    json.append(S_BRACE_C);
    //    if (index != tracks.size() - 1) {
    //        json.append(S_COMMA);
    //    }
    //}
    //
    //json.append(S_BRACE_C);
}

bool lane_seg::Segmenter::isValid() const {
    return valid;
}