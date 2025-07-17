import logging
import os
from flask import Flask, request, jsonify
import random
import librosa
import numpy as np

app = Flask(__name__)

@app.after_request
@app.after_request
 def add_cross_origin_headers(response):
     response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
     response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
     return response



@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics():
    data = request.get_json()
    theme = data.get('theme', '')
    style = data.get('style', 'traditional')
    
    # 加载微调后的彝族儿歌NLP模型（使用实际训练的模型路径）
    from transformers import pipeline
    # 注意：请将模型文件放置在项目根目录的models文件夹下
# 若没有模型，可使用默认模型：model='gpt2'
lyric_generator = pipeline('text-generation', model='gpt2')  # 临时使用默认模型  # 假设模型存储在models目录下的微调版本
    # 验证模型加载状态
    if not lyric_generator: raise RuntimeError('歌词生成模型加载失败')
    
    # 构造更精准的生成提示（结合用户输入+彝族文化关键词）
    user_prompt = data.get('prompt', '')
    cultural_keywords = ['阿依','跳月','火把','索玛花','月琴']  # 彝族文化核心元素
    full_prompt = f'创作{style}风格的彝族儿歌，主题：{theme}，用户提示：{user_prompt}。歌词需包含以下元素（至少3个）：{cultural_keywords}，结构押韵，每段4-6句。歌词开头：'
    
    # 调用模型生成歌词（限制最大长度200字符，避免过长）
    generated = lyric_generator(full_prompt, max_length=200, num_return_sequences=1, pad_token_id=50256)
    lyrics = generated[0]['generated_text'].replace(full_prompt, '').strip()  # 去除生成的前缀提示
    return jsonify({'lyrics': lyrics})

@app.route('/export_midi')
def export_midi():
    # 返回示例MIDI文件
    from io import BytesIO
    midi_data = BytesIO(b'MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0MTrk\x00\x00\x00\x0b\x00\xff\x2f\x00')
    return midi_data.getvalue(), 200, {'Content-Type': 'audio/midi', 'Content-Disposition': 'attachment; filename=composition.mid'}



@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'audio' not in request.files:
        logging.error('请求中未包含音频文件')
        return jsonify({'error': '未提供音频文件'}), 400
    
    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return jsonify({'error': '仅支持WAV/MP3/OGG/FLAC格式'}), 400
    
    try:
        logging.info(f'接收到音频文件: {audio_file.filename}, 大小: {audio_file.content_length}字节')
        
        # 确保临时目录存在
        os.makedirs('temp', exist_ok=True)
        temp_path = os.path.join('temp', 'recording.wav')
        
        audio_file.save(temp_path)
        logging.info(f'音频文件已临时保存到: {temp_path}')
        
        # 验证文件是否成功保存
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise ValueError('音频文件保存失败或为空文件')
            
    except Exception as e:
        logging.error(f'音频文件保存错误: {str(e)}', exc_info=True)
        logging.error(f'文件信息: 文件名={audio_file.filename}, 大小={audio_file.content_length}字节')
        return jsonify({'error': f'音频文件保存失败: {str(e)}'}), 500
    
    try:
        # 音频预处理 - 标准化采样率
        y, sr = librosa.load(temp_path, sr=22050)
        if len(y) == 0:
            logging.error('加载的音频数据为空')
            raise ValueError('音频文件为空或损坏')
            
        # 音频预处理 - 降噪
        y = librosa.effects.preemphasis(y)
        
        # 音频预处理 - 音量归一化
        y = librosa.util.normalize(y)
        
        # 检查音频质量
        if np.max(np.abs(y)) < 0.01:
            logging.error(f'音频信号过弱，最大振幅: {np.max(np.abs(y))}')
            raise ValueError('音频信号过弱')
            
        # 提取音高特征
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        if len(pitches[pitches > 0]) == 0:
            logging.error('无法检测到有效音高')
            raise ValueError('无法检测到有效音高')
            
        pitch_mean = np.mean(pitches[pitches > 0])
        
        # 提取MFCC特征（更能反映音色）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # 使用KNN分类器（基于预训练的彝族音色样本）
        # 示例训练数据（实际应使用真实彝族音色样本训练）
        train_features = np.array([[1.2, -0.5, 0.8], [-0.3, 1.1, -0.2], [0.9, 0.7, -1.0], [-1.5, 0.3, 0.6]])  # 示例MFCC均值
        timbre_types = ['bright', 'warm', 'rich', 'light']
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, timbre_types)
        timbre = knn.predict([mfcc_mean])[0]
        
        # 确定音域范围
        pitch_min = np.min(pitches[pitches > 0])
        pitch_max = np.max(pitches[pitches > 0])
        pitch_range = f"{librosa.hz_to_note(pitch_min)}-{librosa.hz_to_note(pitch_max)}"
        
        return jsonify({
            'timbre': timbre,
            'pitchRange': pitch_range
        })
    except Exception as e:

        logging.error(f'音频分析错误: {str(e)}')
        logging.error(f'音频文件信息: 采样率={sr}, 时长={librosa.get_duration(y=y, sr=sr)}秒')
        logging.error(f'音频特征: 均值={np.mean(y)}, 标准差={np.std(y)}')
        return jsonify({'error': str(e)}), 500

移除了`@app.route('/arrange_chords', methods=['POST'])`路由及其对应的`arrange_chords`函数，该功能相关代码已从服务端彻底删除。
if __name__ == '__main__':
    app.run(port=5000)