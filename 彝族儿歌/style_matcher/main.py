import librosa
import numpy as np
from sklearn.neighbors import NearestNeighbors

class StyleMatcher:
    def __init__(self):
        self.style_features = {
            'pop': self._extract_features('pop'),
            'rock': self._extract_features('rock'),
            'jazz': self._extract_features('jazz')
        }
        self.knn = NearestNeighbors(n_neighbors=1)
        self._train_model()
    
    def _extract_features(self, audio_path=None):
        """使用预设特征值替代缺失的音频样本文件"""
        # 实际应用时应替换为真实音频文件的特征提取
        preset_features = {
            'pop': np.array([-1.2, 0.5, -0.8, 1.1, -0.3, 0.9, -1.0, 0.6, -0.7, 1.2, -0.4, 0.8, -0.5]),
            'rock': np.array([0.3, -1.1, 0.2, -0.9, 1.0, -0.6, 0.7, -1.5, 0.3, -0.6, 1.1, -0.8, 0.4]),
            'jazz': np.array([-0.9, 0.7, -1.0, 0.6, -0.5, 1.2, -0.4, 0.8, -0.5, 0.3, -1.1, 0.2, -0.9])
        }
        return preset_features.get(audio_path, np.zeros(13))
    
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