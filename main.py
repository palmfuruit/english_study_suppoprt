import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import ocr_main
import my_model
import stanza
# import requests
# import nltk
# from nltk.tokenize import sent_tokenize
from googletrans import Translator
import whisper
import tempfile
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# FFmpegのパスを明示的に指定
os.environ["PATH"] += os.pathsep + "D:/FreeTool/ffmpeg-master-latest-win64-lgpl/bin"

# セッションステートの初期化
def initialize_session_state():
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    
    if 'response_data' not in st.session_state:
        st.session_state.response_data = []

    if 'nlp' not in st.session_state:
        # Stanzaの英語モデルをロードしてセッションステートに保存
        st.session_state.nlp = setup_stanza()

    if 'whisper' not in st.session_state:
        # Stanzaの英語モデルをロードしてセッションステートに保存
        st.session_state.whisper = setup_whisper()

    if 'ocr_model' not in st.session_state:
        st.session_state.ocr_model = ocr_main.initialize_ocr()

    if 'translator' not in st.session_state:
        st.session_state.translator = Translator()

    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

# Stanzaのセットアップと文の解析
def setup_stanza():
    # stanza.download('en', verbose=False) # Stanzaの英語モデルをダウンロード
    return stanza.Pipeline('en')  # パイプラインの初期化

def setup_whisper():
    model = whisper.load_model("base")
    return model

# 画像ファイルUpload
def on_file_upload():
    if st.session_state.image_files:
        process_image(st.session_state.image_files)

# 画像ファイルUpload
def on_movie_file_upload():
    if st.session_state.movie_file:
        overWrite = st.empty()
        with overWrite.container():
            st.info("ファイルがアップロードされました。文字起こし中です...")
            transcription = transcribe_audio(st.session_state.whisper, st.session_state.movie_file)
        overWrite.empty()

        # 結果を複数行テキストエリアに表示
        st.text_area("文字起こし結果", value=transcription, height=300)
        st.session_state.uploaded_image = None      # 前回アップロードした画像をクリア
        st.session_state.sentences = split_into_sentences(transcription)        


# 画像の処理
def process_image(image_file):
    overWrite = st.empty()
    with overWrite.container():
        st.info('画像からテキストを読み出し中・・・。')
        original_image = Image.open(image_file)
        st.session_state.sentences = ocr_main.image_to_sentences(np.array(original_image), st.session_state.ocr_model)
    overWrite.empty()
    st.session_state.uploaded_image = original_image

# 音声入力
def transcribe_audio(model, audio_file):
    # 一時ファイルとして音声ファイルを保存
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name
    
    # Whisperで文字起こし
    result = model.transcribe(tmp_file_path, language="en")
    
    # 一時ファイルを削除
    os.remove(tmp_file_path)
    
    return result['text']


# Readingするテキストを選択
def select_text_to_read():
    selected_text = ""
    if st.session_state.sentences:
        with st.sidebar:
            if st.session_state.uploaded_image: 
                st.image(st.session_state.uploaded_image, use_column_width=True)
            
            grammar_labels_with_counts = get_grammar_label_with_counts()
            selected_grammar = st.selectbox('使用している文法でフィルタ', grammar_labels_with_counts, index=None)
            if selected_grammar:
                selected_grammar = selected_grammar.split(' (')[0]

            if selected_grammar == None:
                selected_text = st.radio("文を選択してください。", st.session_state.sentences)
            else:
                filtered_sentences = []
                for i, preds in enumerate(st.session_state.response_data):
                    pred_labels = [grammer_labels[idx] for idx, label in enumerate(preds) if label == 1.0]
                    if selected_grammar in pred_labels:
                        filtered_sentences.append(st.session_state.sentences[i])
                
                if filtered_sentences:
                    # Display only filtered sentences
                    selected_text = st.radio("文を選択してください。", filtered_sentences)
                else:
                    # Show a message if no sentences match the selected grammar
                    st.write("選択された文法に一致する文がありません。")    
    
    return selected_text


### 文章の分割
from textblob import TextBlob

def split_into_sentences(text):
    blob = TextBlob(text)
    sentence_list = blob.sentences
    sentence_list = [str(sentence) for sentence in sentence_list]

    return sentence_list

def get_subtree_span(token, sentence):
    start = token.start_char
    end = token.end_char

    # トークンの子供(動詞と句読点除く)を探し、その範囲を確認する
    for word in sentence.words:
        if word.head == token.id and (word.upos not in ['VERB', 'PUNCT']) and (word.deprel not in ['appos', 'conj', 'advmod']):  # トークンが現在の単語の親である場合
            # 子トークンの範囲を確認し、現在の範囲と比較して更新する
            start = min(start, word.start_char)
            end = max(end, word.end_char)

    return start, end

@st.cache_data
def get_nlp_doc(sentence):
    return st.session_state.nlp(sentence)


def get_span_color(span_type):
    colors = {
        'subject': 'blue',
        'verb': 'red',
        'auxiliary': 'pink',
        'object': 'yellowgreen',
        'indirect_object': 'green',
        'complement': 'orange',
        'objective_complement': 'brown',
    }
    return colors.get(span_type, 'black')

# 下線スタイルを適用する関数
def apply_underline(text, color):
    return f"<span style='border-bottom: 2pt solid {color}; position: relative;'>{text}</span>"

def expand_span(word, parent_word, sentence):
    start_idx = word.start_char
    end_idx = word.end_char
    
    # 名詞が対象の場合、冠詞や所有格のチェックを行う
    if word.pos in ['NOUN', 'PRON','PROPN']:
        for related_word in sentence.words:
            if(related_word.head == word.id) and (related_word.start_char is not None) and \
            ((related_word.pos == 'DET') or (related_word.deprel == 'nmod:poss') or (related_word.deprel == 'nummod') or (related_word.deprel == 'nmod') or (related_word.deprel == 'amod')):
                if related_word.start_char < start_idx:
                    start_idx = related_word.start_char
                if related_word.end_char > end_idx:
                    end_idx = related_word.end_char 
    
    # 親が節の場合 子の範囲も下線を引く
    if (word.deprel in ["xcomp", "ccomp"]) or \
       (word.deprel in ["conj", "appos"] and parent_word.deprel in ["xcomp", "ccomp"]):
        for related_word in sentence.words:
            if(related_word.head == word.id) and (related_word.start_char is not None) and (related_word.deprel not in ["advcl"]):
                if related_word.start_char < start_idx:
                    start_idx = related_word.start_char
                if related_word.end_char > end_idx:
                    end_idx = related_word.end_char 
    
    # 子がcompound, flatの場合は範囲を拡張
    for related_word in sentence.words:
        if(related_word.head == word.id) and (related_word.deprel in['compound', 'flat']) and (related_word.start_char is not None):
            if related_word.start_char < start_idx:
                start_idx = related_word.start_char
            if related_word.end_char > end_idx:
                end_idx = related_word.end_char
                
    return start_idx, end_idx

def extract_target_tokens(doc):
    target_tokens = []
    
    for sentence in doc.sentences:
        root_word = next((word for word in sentence.words if word.head == 0), None)
        root_id = root_word.id 

        for i, word in enumerate(sentence.words):
            span_type = None
            parent_word = sentence.words[word.head - 1] if word.head > 0 else None

            ## ROOTトークン
            if (word.head == 0) or (word.deprel == "conj" and parent_word.id == root_id): 
                if (word.pos == 'VERB') or \
                   ((word.deprel == 'compound:prt') and (word.head == root_id) and (root_word.pos == 'VERB')):  # 句動詞
                    span_type = "verb"
                elif (word.pos != 'VERB'):
                    span_type = "complement"
            
            ## 親がROOTトークン
            elif (word.head == root_id) or \
                 ((word.deprel == "conj" or parent_word.deprel == "conj") and parent_word.head == root_id): 
                if (word.pos == 'AUX') or (parent_word.pos == 'AUX'):
                    span_type = "auxiliary"
                elif ("subj" in word.deprel) or ("subj" in parent_word.deprel):
                    span_type = "subject"
                elif (word.deprel in ["obj"]) or (parent_word.deprel in ["obj"]):
                    span_type = "object"
                elif (word.deprel in ["iobj"]) or (parent_word.deprel in ["iobj"]):
                    span_type = "indirect_object"
                elif (parent_word.pos == 'VERB') and (word.deprel == "ccomp" or parent_word.deprel == "ccomp"):
                    span_type = "object"
                elif (parent_word.pos == 'VERB') and (word.deprel == "xcomp" or parent_word.deprel == "xcomp"):
                    other_obj_exists = any((w.deprel in ["obj", "iobj"]) and (w.head == root_id) and (root_word.pos == 'VERB') for w in sentence.words)
                    if other_obj_exists:
                        span_type = "objective_complement"
                    else:
                        if root_word.lemma in ['become', 'seem', 'be', 'appear']:
                            span_type = "complement"
                        else:
                            span_type = "object"
                elif (word.deprel in ["xcomp"]) and (word.pos in ['NOUN', 'PRON','PROPN', 'ADJ']):
                    span_type = "objective_complement"


                
            if span_type:
                start_idx, end_idx = expand_span(word, root_word, sentence)

                color = get_span_color(span_type)
                target_tokens.append({
                    "text": word.text,  
                    "start_idx": start_idx,
                    "end_idx":  end_idx,
                    "type": span_type,
                    "color": color
                })
                
    return target_tokens




def apply_underline_to_text(text, target_tokens):
    underlined_text = list(text)  # 文字列をリストに変換して操作しやすくする
    
    for token in target_tokens:
        start_idx = token["start_idx"]
        end_idx = token["end_idx"]
        color = token["color"]

        underlined_text[start_idx] = apply_underline(''.join(underlined_text[start_idx:end_idx]), color)
        for i in range(start_idx + 1, end_idx):
            underlined_text[i] = ''  # 一度適用した部分を消す

    return ''.join(underlined_text)


# 主語、動詞、目的語、補語に下線を引く関数
def underline_clauses(text, doc):
    target_tokens = extract_target_tokens(doc)
    return apply_underline_to_text(text, target_tokens)


    
def determine_sentence_pattern(spans):
    has_subject = False
    has_object = False
    has_complement = False
    has_object_complement = False
    has_indirect_object = False

    for span, span_type in spans:
        if span_type == 'subject':
            has_subject = True
        elif span_type == 'direct_object':
            has_object = True
        elif span_type == 'indirect_object':
            has_indirect_object = True
        elif span_type == 'complement':
            has_complement = True

    if has_subject and not has_object and not has_complement:
        return "第1文型 (SV)"
    elif has_subject and has_complement and not has_object:
        return "第2文型 (SVC)"
    elif has_subject and has_object and has_indirect_object:
        return "第4文型 (SVOO)"
    elif has_subject and has_object and has_complement:
        return "第5文型 (SVOC)"
    elif has_subject and has_object:
        return "第3文型 (SVO)"
    else:
        return ""


grammer_labels = [
    '受動態',
    '完了形',
    '比較',
    '仮定法',
    '使役',
    'WH名詞節',
]

# 文法ラベルにそれぞれの文法に適合している文の数を追加する関数
def get_grammar_label_with_counts():
    label_counts = {label: 0 for label in grammer_labels}
    
    for preds in st.session_state.response_data:
        for idx, label in enumerate(preds):
            if label == 1.0:
                label_counts[grammer_labels[idx]] += 1
    
    labeled_grammar_labels = [f"{label} ({count})" for label, count in label_counts.items()]
    return labeled_grammar_labels

@st.cache_data
def predict_grammer_label(sentences):

    print('predict_grammer -- start --')

    results = my_model.predict_labels(sentences)
    if not results:
        print("分類結果を取得できません。")
    
    print('predict_grammer -- end --')
    return results

@st.cache_data
# 文が該当する文法を表示 (仮定法、比較級、・・・)
def sentence_to_grammer_label(selected_text):
    selected_index = st.session_state.sentences.index(selected_text)
    preds = st.session_state.response_data[selected_index]
    pred_labels = [grammer_labels[idx] for idx, label in enumerate(preds) if label == 1.0]
    pred_labels_html = ""
    for label in pred_labels:
        pred_labels_html += f"<span style='background-color: pink; padding: 2px 4px; margin-right: 5px;'>{label}</span>"

    return pred_labels_html

@st.cache_data
def translate(en_text):
    translated_obj = st.session_state.translator.translate(en_text, dest="ja")
    return translated_obj.text


def get_token_info(doc):
    tokens_info = []

    for sentence in doc.sentences:
        for word in sentence.words:
            token_info = {
                'Text': word.text,
                'Lemma': word.lemma,
                'POS': word.upos,
                # 'XPOS': word.xpos,      # 言語固有の品詞タグ
                'Dependency': word.deprel,
                'Head': sentence.words[word.head - 1].text if 0 < word.head else 'ROOT',
                # 'Children': [child.text for child in sentence.words if child.head == word.id],
                # 'Feats': word.feats,
                # 'Deps': word.deps,
                # 'ID': word.id,
                # 'Misc': word.misc
            }
            tokens_info.append(token_info)

    
    tokens_df = pd.DataFrame(tokens_info)

    return tokens_df



@st.cache_data
def display_legend():
    legend_html = """
    <div style='border: 2px solid black; padding: 10px; margin-bottom: 20px;'>
        <p><span style='border-bottom: 2px solid blue; display: inline-block; width: 80px;'></span> 主語(Subject)</p>
        <p><span style='border-bottom: 2px solid red; display: inline-block; width: 80px;'></span> 動詞 (Verb)</p>
        <p><span style='border-bottom: 2px solid pink; display: inline-block; width: 80px;'></span> 助動詞 (Auxiliary)</p>
        <p><span style='border-bottom: 2px solid yellowgreen; display: inline-block; width: 80px;'></span> 目的語 (Object)</p>
        <p><span style='border-bottom: 2px solid green; display: inline-block; width: 80px;'></span> 間接目的語 (Indirect Object)</p>
        <p><span style='border-bottom: 2px solid orange; display: inline-block; width: 80px;'></span> 補語 (Complement)</p>
        <p><span style='border-bottom: 2px solid brown; display: inline-block; width: 80px;'></span> 目的語補語 (Objective Complement)</p>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)



# メイン関数
def main():
    initialize_session_state()

    st.title('英語Reading学習アプリ')

    # ラジオボタンで「テキスト」か「画像」を選択
    input_type = st.radio("英語テキストの入力方法", ("テキスト", "画像", "動画"))

    if input_type == "画像":
        # 画像のアップロードとOCR処理
        image_files = st.file_uploader('英語のテキストが記載された画像を選択', type=['jpg', 'jpeg', 'png'],
                                        accept_multiple_files=False, on_change=on_file_upload, key='image_files')
                                        
    elif input_type == "テキスト":
        # テキストボックスと解析ボタンを表示
        text_input = st.text_area("英語のテキストを入力してください:", height=300)
        if st.button("入力"):
            # 入力されたテキストをStanzaで文に分割して保持
            print('====テキストを文章に分割 (start)')
            st.session_state.uploaded_image = None      # 前回アップロードした画像をクリア
            st.session_state.sentences = split_into_sentences(text_input)
            print('====テキストを文章に分割 (end)')
    
    elif input_type == "動画":
        # ファイルアップロード
        movie_file = st.file_uploader("MP4ファイルをアップロードしてください", type=["mp4"], on_change=on_movie_file_upload, key='movie_file')


    if st.session_state.sentences:
        overWrite = st.empty()
        with overWrite.container():
            st.info('各文章を文法で分類中・・・')
            st.session_state.response_data = predict_grammer_label(st.session_state.sentences)
        overWrite.empty()

    # 文の選択
    selected_text = select_text_to_read()
            

    st.divider() # 水平線
    if selected_text:
        # 英文が該当する文法を表示 (仮定法、比較級、・・・)        
        pred_labels_html = sentence_to_grammer_label(selected_text)
        st.write(pred_labels_html, unsafe_allow_html=True)
        
        overWrite = st.empty()
        with overWrite.container():
            st.info('選択した文に下線を引いています・・・')
            doc = get_nlp_doc(selected_text)
            main_clause_sentence = underline_clauses(selected_text, doc)
        overWrite.empty()

        st.markdown(main_clause_sentence, unsafe_allow_html=True)

        # if st.checkbox("下線を表示"):
        #     st.markdown(main_clause_sentence, unsafe_allow_html=True)
        # else:
        #     st.write(selected_text)

        # # 文型
        # spans = extract_spans(doc)
        # sentence_pattern = determine_sentence_pattern(spans)
        # st.write(sentence_pattern)
        

        if st.checkbox("翻訳文を表示"):
            translated_text = translate(selected_text)
            st.write(translated_text)
        
        # トークン情報の表を出力 (開発用)
        # token_df = get_token_info(doc)
        # st.dataframe(token_df, width=1200)

    # 凡例を表示
    display_legend()

if __name__ == "__main__":
    main()