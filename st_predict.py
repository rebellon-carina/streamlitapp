import streamlit as st
import joblib

import soundfile as sf
import io
from six.moves.urllib.request import urlopen

@st.cache(allow_output_mutation=True)
### ALL FUNCTION DEFINITIONS HERE
def librosa_features(data, sample_rate):
    
    '''this will create an array of some librosa features
    (e.g. zero_crossing_rate, root mean square energy, spectral centroid and rollfoof,
        chroma_stft, mfcc, rms, melspectrogram)
    concatenated for each audio'''
    
    
    result = np.array([])
    
   
    # Root Mean Square Value (one column, the average )
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # add as column 
    
     # Spectral Centroid (one column, the average )
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_centroid)) # stacking horizontally
    
        
     # Spectral Rolloff (one column, the average )
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_rolloff)) # add as column 
    
        
    
    # Zero Crossing Rate (one column, the average )
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # add as column 
    

    
    # Chroma_stft (12 columns)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    
    # MFCC  (40 columns)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
      
    
    # MelSpectogram (128 columns)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_basic_features(audio, sfreq):
    ''' get librosa features only for the test dataset '''
        
    #Trim leading and trailing silence from an audio signal.
    yt, index = librosa.effects.trim(audio)
    
    
    #librosa features
    res1 = librosa_features(yt, sfreq)

    return res1


def predict(audio, sfreq):
    
    librosa_X = []
    librosa_X.append(get_basic_features(audio, sfreq))
    df_feature = pd.DataFrame(librosa_X)

    #scale
    x_scaled = sc.transform(pd.DataFrame(df_feature))
    
    #predict
    y_pred = loaded_model.predict(x_scaled)
    
    #return the prediction
    return le.inverse_transform(y_pred)


### END FUNCTIONS


loaded_model = joblib.load("./model/vote_model.bin")
sc = joblib.load('./model/vote_scaler.bin')
le = joblib.load('./model/vote_le.bin')

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Speech Emotion Recognition")
st.text("Provide URL of audio file (.wav format)")

url = 'https://github.com/rebellon-carina/Capstone---Speech-Emotion-Recognition/raw/master/test_audio/kyle_happy.wav'
path = st.text_input('Enter Audio URL to Classify.. ', url)

if path is not None:
    audio, sfreq = sf.read(io.BytesIO(urlopen(path).read()))
    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
        
        label =predict(audio, sfreq)
        st.write(classes[label[0]])
        
    