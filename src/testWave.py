from src.modelFetch import modelImageFetch

def testWaveImage(link):
    classes_x = modelImageFetch(link, "cnn_wave.h5", 256, 512)

    return classes_x