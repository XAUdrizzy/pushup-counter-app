import { StatusBar } from "expo-status-bar";
import { StyleSheet, Text, View, Dimensions, Platform } from "react-native";
import React, { useEffect, useRef, useState } from "react";
import Svg, { Circle, Line } from "react-native-svg";

import { Camera } from "expo-camera";
import * as ScreenOrientation from "expo-screen-orientation";
import { CameraType } from "expo-camera/build/Camera.types";
import { ExpoWebGLRenderingContext } from "expo-gl";

import * as tf from "@tensorflow/tfjs";
import * as posedetection from "@tensorflow-models/pose-detection";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
import * as posenet from "@tensorflow-models/posenet";

const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === "android";
const IS_IOS = Platform.OS === "ios";

// Set camera preview size
// To render without distortion use 16:9 ratio for iOS
// and 4:3 ratio for android
const CAM_PREVIEW_WIDTH = Dimensions.get("window").width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Score threshold for pose detection results
const MIN_KEYPOINT_SCORE = 0.3;

// Size of the resized output from TensorCamera
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Auto-redener TensorCamera preview
const AUTO_RENDER = true;

// Load model from app bundle (true) or through network (false)
const LOAD_MODEL_FROM_BUNDLE = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [cameraType, setCameraType] = useState<CameraType>(CameraType.front);
  const [debugMode, setDebugMode] = useState(true);
  // Using 'useRef' so that changing it won't trigger a re-render
  // - null: unset (initial value)
  // - 0: animation frame/loop has been canceled
  // - >0: animation frame has been scheduled
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize
      await tf.ready();

      // Load movenet model
      const movenetModelConfig: posedetection.MoveNetModelConfig = {
        modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      };

      // If model is stored on local device
      /* 
      if (LOAD_MODEL_FROM_BUNDLE) {
        const modelJson = require("./model.json");
        const modelWeights1 = require("./group1-shard1of2.bin");
        const modelWeights2 = require("./group1-shard2of2.bin");
        movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
          modelWeights1,
          modelWeights2,
        ]);
      }
      */

      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        movenetModelConfig
      );

      setModel(model);

      // Tensorflow is ready!
      setTfReady(true);
    }
    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      const imageTensor = tf.reverse(images.next().value as tf.Tensor3D,1);

      const startTs = Date.now();
      const poses = await model!.estimatePoses(imageTensor);
      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      setPoses(poses);
      tf.dispose([imageTensor]);

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      rafId.current = requestAnimationFrame(loop);
    };
    loop();
  };

  const renderKeypoints = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android or when using back camera on iOS
          const flipX = IS_ANDROID || cameraType === CameraType.back;
          const x = flipX ? getOutputTensorWidth() - k.x : k.x;
          const y = k.y;
          const cx = translateX(x);
          const cy = translateY(y);
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={cx}
              cy={cy}
              r="4"
              strokeWidth="2"
              fill="#00AA00"
              stroke="white"
            />
          );
        });
      //console.log(keypoints);
      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderSkeleton = (minConfidence = MIN_KEYPOINT_SCORE) => {
    const skeleton: React.ReactElement[] = [];
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints;
      posedetection.util
        .getAdjacentPairs(posedetection.SupportedModels.MoveNet)
        .forEach(([i, j]) => {
          const kp1 = keypoints[i];
          const kp2 = keypoints[j];

          // If score is null just show the keypoint
          const score1 = kp1.score != null ? kp1.score : 1;
          const score2 = kp2.score != null ? kp2.score : 1;

          if (score1 >= minConfidence && score2 >= minConfidence) {
            const cx1 = translateX(kp1.x);
            const cy1 = translateY(kp1.y);
            const cx2 = translateX(kp2.x);
            const cy2 = translateY(kp2.y);
            skeleton.push(
              <Line
                key={`skeletonline_${kp1.name}-${kp2.name}`}
                x1={cx1}
                x2={cx2}
                y1={cy1}
                y2={cy2}
                stroke="red"
                strokeWidth="2"
              />
            );
          }
        });
    }
    return <Svg style={styles.svg}>{skeleton}</Svg>;
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to {cameraType === CameraType.front ? "back" : "front"}
        </Text>
      </View>
    );
  };

  const renderDebugModeSwitcher = () => {
    return (
      <View style={styles.debugModeSwitcher} onTouchEnd={handleSwitchDebugMode}>
        <Text>Debug {debugMode ? "ON" : "OFF"}</Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === CameraType.front) {
      setCameraType(CameraType.back);
    } else {
      setCameraType(CameraType.front);
    }
  };

  const handleSwitchDebugMode = () => {
    if (debugMode) {
      setDebugMode(false);
    } else {
      setDebugMode(true);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const translateX = (x: number): number => {
    return (x / getOutputTensorWidth()) *
    (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
  }

  const translateY = (y: number): number => {
    return (y / getOutputTensorHeight()) *
    (isPortrait() ? CAM_PREVIEW_HEIGHT: CAM_PREVIEW_WIDTH);
 
  }

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor
    // to get a better result. Without this it would stretch the image too much
    // Same for getOutputTensorHeight
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    // On iOS landscape mode, switch width and height of the output tensor
    // to get a better result. Without this it would stretch the image too much
    // Same for getOutputTensorHeight
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On android, the camera texture rotates behind the scene when the phone
    // changes orientation so we don't need to rotate it in TensorCamera
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS the camera texture won't rotate automatically.
    // Calculate rotation angles to rotate TensorCamera internally
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === CameraType.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === CameraType.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    // No need to specify 'cameraTextureWidth' and
    // 'cameraTextureHeight in TensorCamera props
    return (
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={cameraType}
          // tensor related props
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream}
          useCustomShadersToResize={false}
          cameraTextureWidth={CAM_PREVIEW_WIDTH}
          cameraTextureHeight={CAM_PREVIEW_HEIGHT}
          resizeDepth={3}
        />

        {debugMode && renderKeypoints()}
        {debugMode && renderSkeleton()}

        {renderFps()}
        {renderDebugModeSwitcher()}
        {renderCameraTypeSwitcher()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: "relative",
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: "relative",
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: "absolute",
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  svg: {
    width: "100%",
    height: "100%",
    position: "absolute",
    zIndex: 30,
  },
  fpsContainer: {
    position: "absolute",
    top: 10,
    left: 10,
    width: 80,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 100,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  debugModeSwitcher: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 160,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
