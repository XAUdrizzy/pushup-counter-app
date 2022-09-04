import CameraView from './components/CameraView'
import { View, StyleSheet } from 'react-native'

export default function App() {

  return (
    <View style={styles.container}>
      <CameraView />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignContent: "center",
  },
});
