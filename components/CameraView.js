import { StyleSheet, Text, View, TouchableOpacity, Button } from "react-native";
import { useEffect, useState, useRef } from "react";
import { Camera, CameraType } from "expo-camera";
import * as tf from "@tensorflow/tfjs"
import * as posenet from "@tensorflow-models/posenet"
import Expo2DContext from "expo-2d-context";
import { GLView } from "expo-gl";

//TODO: Canvas por cima da cameraview

export default function CameraView() {
  const [type, setType] = useState(CameraType.back);
  const [hasPermission, setHasPermission] = Camera.useCameraPermissions();

  const cameraRef = useRef(null);
  const canvasRef = useRef(null);

  function toggleCameraType() {
    setType((current) =>
      current === CameraType.back ? CameraType.front : CameraType.back
    );
  }

  _onGLContextCreate = (g1) => {
    var ctx = new Expo2DContext(g1);
    ctx.fillStyle = "red";
    ctx.fillRect(20, 40, 100, 100);
    ctx.stroke();
    ctx.flush();
  }

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={toggleCameraType}
          >
            <Text style={styles.text}> Flip </Text>
          </TouchableOpacity>
        </View>
      </Camera>
      <GLView style={styles.canvas} onContextCreate={this._onGLContextCreate} />
    </View>
  );

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignContent: "center",
  },
  camera: {
    flex: 1,
    zIndex: 10,
  },
  canvas: {
    width: '50%',
    height: '100%',
    zIndex: 9,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: "transparent",
    flexDirection: "row",
    margin: 20,
  },
  button: {
    flex: 0.1,
    alignSelf: "flex-end",
    alignItems: "center",
  },
  text: {
    fontSize: 18,
    color: "white",
    marginBottom: 10,
  },
});
