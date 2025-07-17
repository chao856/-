import librosa
import numpy as np
from sklearn.neighbors import NearestNeighbors

class StyleMatcher:
    def __init__(self):
        self.style_features = {
            'pop': self._extract_features('pop_sample.wav'),
            'rock': self._extract_features('rock_sample.wav'),
            'jazz': self._extract_features('jazz_sample.wav')
        }
        self.knn = NearestNeighbors(n_neighbors=1)
        self._train_model()
    
    def _extract_features(self, audio_path):
        """提取音频特征"""
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return np.mean(mfcc, axis=1)
    
    def _train_model(self):
        """训练KNN模型"""
        X = np.array(list(self.style_features.values()))
        self.knn.fit(X)
    
    def match_style(self, audio_path):
        """匹配音乐风格"""
        features = self._extract_features(audio_path)
        distances, indices = self.knn.kneighbors([features])
        return list(self.style_features.keys())[indices[0][0]]

if __name__ == '__main__':
    matcher = StyleMatcher()
    print(matcher.match_style('test_audio.wav'))