# app.py - Démo Streamlit pour la classification de sentiments
# RNN vs LSTM sur le dataset IMDb

import os
import re
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# Désactivation des warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

# Configuration de la page
st.set_page_config(
    page_title="Analyse de sentiments IMDb - RNN vs LSTM",
    page_icon="🎬",
    layout="wide"
)

# ==================== DÉFINITION DES MODÈLES ====================
VOCAB_SIZE = 20000
EMBED_DIM = 128
HIDDEN_DIM = 256
MAX_LEN = 200

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim,
                          num_layers=2, batch_first=True,
                          dropout=0.3, bidirectional=True,
                          nonlinearity="tanh")
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)

# ==================== CHARGEMENT DES MODÈLES ====================
@st.cache_resource
def load_models():
    """Charge les modèles et le vocabulaire"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rnn_model = RNNModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, 1).to(device)
    lstm_model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, 1).to(device)
    
    models_ok = True
    errors = []
    
    if os.path.exists("data/rnn_model.pth"):
        rnn_model.load_state_dict(torch.load("data/rnn_model.pth", map_location=device))
        rnn_model.eval()
    else:
        models_ok = False
        errors.append("rnn_model.pth")
    
    if os.path.exists("data/lstm_model.pth"):
        lstm_model.load_state_dict(torch.load("data/lstm_model.pth", map_location=device))
        lstm_model.eval()
    else:
        models_ok = False
        errors.append("lstm_model.pth")
    
    vocab = None
    if os.path.exists("data/vocab.pkl"):
        with open("data/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    else:
        models_ok = False
        errors.append("vocab.pkl")
    
    return rnn_model, lstm_model, vocab, device, models_ok, errors

# ==================== FONCTIONS DE BASE ====================
def clean_text(text):
    """Nettoie le texte comme dans l'entraînement"""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text, model, vocab, device, max_len=MAX_LEN):
    """Prédit le sentiment d'un texte"""
    text_clean = clean_text(text)
    words = text_clean.split()[:max_len]
    unk_idx = vocab.get('<UNK>', 1)
    sequence = [vocab.get(word, unk_idx) for word in words]
    pad_idx = vocab.get('<PAD>', 0)
    sequence += [pad_idx] * (max_len - len(sequence))
    tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output).item()
    
    return probability

def predict_with_uncertainty(text, model, vocab, device, n_passes=10):
    """Estimation de l'incertitude par Monte Carlo Dropout"""
    model.train()
    probabilities = []
    
    for _ in range(n_passes):
        prob = predict_sentiment(text, model, vocab, device)
        probabilities.append(prob)
    
    model.eval()
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    
    return mean_prob, std_prob

def detect_language(text):
    """Détection automatique de la langue"""
    try:
        from langdetect import detect
        return detect(text)
    except ImportError:
        return "unknown"
    except:
        return "unknown"

def detect_sarcasm(text):
    """Détection basique du sarcasme/ironie"""
    sarcasm_indicators = ['yeah right', 'as if', 'sure', 'obviously', 'clearly', 'whatever']
    exclamation_count = text.count('!')
    quote_count = text.count('"')
    question_count = text.count('?')
    
    score = 0
    for indicator in sarcasm_indicators:
        if indicator in text.lower():
            score += 0.25
    
    if exclamation_count > 2 and quote_count > 0:
        score += 0.2
    if question_count > 2 and exclamation_count > 1:
        score += 0.15
    
    return min(score, 1.0)

def analyze_sentence_level(text, model, vocab, device):
    """Analyse phrase par phrase du sentiment"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
    except ImportError:
        sentences = text.split('.')
    else:
        sentences = sent_tokenize(text)
    
    results = []
    for i, sent in enumerate(sentences[:10]):
        if len(sent.strip()) > 10:
            prob = predict_sentiment(sent, model, vocab, device)
            results.append({
                "sentence": sent.strip()[:80],
                "probability": prob,
                "index": i+1
            })
    return results

def suggest_improvements(text, prob):
    """Suggère des améliorations du texte"""
    words = clean_text(text).split()
    improvements = []
    
    positive_words = ['amazing', 'excellent', 'fantastic', 'brilliant', 'perfect', 'love', 'great', 'wonderful']
    negative_words = ['terrible', 'awful', 'boring', 'waste', 'bad', 'hate', 'disappointing', 'worst']
    
    for word in words:
        if word in negative_words and prob < 0.5:
            improvements.append(f"Remplacer '{word}' par un terme plus positif (ex: 'good', 'great')")
        elif word in negative_words and prob > 0.5:
            improvements.append(f"Le mot '{word}' est négatif alors que la critique est positive → incohérence potentielle")
    
    if prob < 0.3:
        improvements.append("Ajouter des adjectifs positifs comme 'amazing', 'excellent'")
    elif prob > 0.7:
        improvements.append("Le texte est déjà très positif, bonne rédaction !")
    
    return list(set(improvements))[:5]

def optimal_length_analysis(text, model, vocab, device):
    """Analyse l'impact de la longueur sur la prédiction"""
    words = clean_text(text).split()
    if len(words) < 20:
        return []
    
    results = []
    for length in [50, 100, 150, 200]:
        if length <= len(words):
            truncated = " ".join(words[:length])
            prob = predict_sentiment(truncated, model, vocab, device)
            results.append({"length": length, "probability": prob})
    
    return results

def get_word_importance(text, model, vocab, device, max_len=MAX_LEN):
    """Analyse simplifiée des mots importants"""
    text_clean = clean_text(text)
    words = text_clean.split()[:max_len]
    
    if len(words) == 0:
        return []
    
    base_prob = predict_sentiment(text, model, vocab, device, max_len)
    word_importance = []
    
    for i, word in enumerate(words[:30]):
        modified_words = words[:i] + words[i+1:]
        modified_text = " ".join(modified_words)
        if modified_text.strip():
            new_prob = predict_sentiment(modified_text, model, vocab, device, max_len)
            importance = abs(new_prob - base_prob)
            word_importance.append((word, importance))
    
    return sorted(word_importance, key=lambda x: x[1], reverse=True)[:10]

def mask_word_importance(text, word_to_mask, model, vocab, device, max_len=MAX_LEN):
    """Calcule l'impact de la suppression d'un mot"""
    base_prob = predict_sentiment(text, model, vocab, device, max_len)
    modified_text = re.sub(rf'\b{word_to_mask}\b', '[MASQUÉ]', text, flags=re.IGNORECASE)
    new_prob = predict_sentiment(modified_text, model, vocab, device, max_len)
    return base_prob, new_prob, abs(new_prob - base_prob)

def analyze_batch(texts, model, vocab, device, max_len=MAX_LEN):
    """Analyse plusieurs critiques en batch"""
    results = []
    for i, text in enumerate(texts):
        if text.strip():
            prob = predict_sentiment(text, model, vocab, device, max_len)
            sentiment = "POSITIF" if prob > 0.5 else "NÉGATIF"
            results.append({
                "id": i+1,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "probability": prob,
                "sentiment": sentiment
            })
    return results

# ==================== FONCTIONS D'INTERPRÉTATION ====================
def explain_prediction(text, prob, important_words, threshold=0.5):
    """Génère une explication textuelle de la prédiction"""
    
    if prob > threshold:
        sentiment = "POSITIF"
        emoji = "😊"
        if prob > 0.8:
            intensite = "très confiant"
        elif prob > 0.6:
            intensite = "modérément confiant"
        else:
            intensite = "légèrement confiant"
    else:
        sentiment = "NÉGATIF"
        emoji = "😞"
        if prob < 0.2:
            intensite = "très confiant"
        elif prob < 0.4:
            intensite = "modérément confiant"
        else:
            intensite = "légèrement confiant"
    
    # Mots clés influents
    pos_words = [w for w, imp in important_words[:5] if imp > 0.02]
    neg_words = [w for w, imp in important_words[:5] if imp > 0.02]
    
    explanation = f"""
    ### 📋 Explication de la prédiction
    
    **Sentiment prédit :** {sentiment} {emoji}
    **Niveau de confiance :** {intensite} ({prob:.1%})
    
    **Facteurs déterminants :**
    """
    
    if prob > threshold:
        explanation += f"\n- Mots positifs détectés : {', '.join(pos_words[:5]) if pos_words else 'aucun mot fortement positif'}"
        explanation += "\n- Ces mots contribuent à orienter la critique vers un avis favorable."
    else:
        explanation += f"\n- Mots négatifs détectés : {', '.join(neg_words[:5]) if neg_words else 'aucun mot fortement négatif'}"
        explanation += "\n- Ces mots contribuent à orienter la critique vers un avis défavorable."
    
    if len(pos_words) > 0 and len(neg_words) > 0:
        explanation += "\n\n⚠️ **Note :** La critique contient à la fois des mots positifs et négatifs, ce qui peut indiquer un avis mitigé."
    
    return explanation

def plot_word_importance(words_importance, prob, title="Impact des mots sur la prédiction"):
    """Affiche un graphique des mots les plus influents"""
    
    if not words_importance:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    words = [w for w, imp in words_importance[:10]]
    impacts = [imp for w, imp in words_importance[:10]]
    
    colors = ['#e74c3c' if imp > 0 else '#2ecc71' for imp in impacts]
    bars = ax.barh(words, impacts, color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Impact sur la prédiction')
    ax.set_title(f'{title}\nProbabilité finale : {prob:.1%}')
    
    for bar, imp in zip(bars, impacts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.2%}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def compare_predictions_with_explanation(prob_rnn, prob_lstm, words_rnn, words_lstm, threshold=0.5):
    """Compare les deux modèles avec explications"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Explication RNN")
        if prob_rnn > threshold:
            st.success(f"**Prédiction : POSITIF ({prob_rnn:.1%})**")
        else:
            st.error(f"**Prédiction : NÉGATIF ({prob_rnn:.1%})**")
        
        st.markdown("**Mots clés identifiés :**")
        for word, imp in words_rnn[:5]:
            st.markdown(f"- `{word}` : impact {imp:.2%}")
    
    with col2:
        st.markdown("### 🔵 Explication LSTM")
        if prob_lstm > threshold:
            st.success(f"**Prédiction : POSITIF ({prob_lstm:.1%})**")
        else:
            st.error(f"**Prédiction : NÉGATIF ({prob_lstm:.1%})**")
        
        st.markdown("**Mots clés identifiés :**")
        for word, imp in words_lstm[:5]:
            st.markdown(f"- `{word}` : impact {imp:.2%}")
    
    # Analyse du désaccord
    if (prob_rnn > threshold) != (prob_lstm > threshold):
        st.warning("""
        ### ⚠️ Désaccord entre les modèles
        
        **Pourquoi ?**
        - Le RNN est plus sensible aux mots en début/fin de phrase
        - Le LSTM capture mieux le contexte sur l'ensemble du texte
        - La critique contient probablement des nuances (opinions mitigées)
        
        **Recommandation :** Le LSTM (84.30% accuracy) est généralement plus fiable.
        """)

def check_text_coherence(text, prob, important_words):
    """Vérifie la cohérence entre le sentiment et les mots utilisés"""
    
    positive_words = ['amazing', 'excellent', 'fantastic', 'great', 'love', 'perfect', 'wonderful', 'brilliant']
    negative_words = ['terrible', 'awful', 'boring', 'waste', 'hate', 'disappointing', 'worst', 'bad']
    
    text_lower = text.lower()
    
    found_positive = [w for w in positive_words if w in text_lower]
    found_negative = [w for w in negative_words if w in text_lower]
    
    inconsistencies = []
    
    # Cas 1 : sentiment positif mais mots négatifs
    if prob > 0.6 and found_negative:
        inconsistencies.append(f"Sentiment positif mais présence de mots négatifs : {', '.join(found_negative[:3])}")
    
    # Cas 2 : sentiment négatif mais mots positifs
    if prob < 0.4 and found_positive:
        inconsistencies.append(f"Sentiment négatif mais présence de mots positifs : {', '.join(found_positive[:3])}")
    
    # Cas 3 : présence de négations
    if 'not' in text_lower or "n't" in text_lower:
        inconsistencies.append("Présence de négations détectée ('not', 'n't') → peut inverser le sens des adjectifs")
    
    # Cas 4 : présence de sarcasme
    sarcasm_score = detect_sarcasm(text)
    if sarcasm_score > 0.3:
        inconsistencies.append(f"Sarcasme potentiel détecté ({sarcasm_score:.0%}) → la prédiction peut être inversée")
    
    return inconsistencies

def display_interpretation_tab(model, vocab, device, threshold):
    """Affiche l'onglet d'interprétation"""
    
    st.subheader("🔍 Interprétation intelligente des résultats")
    st.markdown("Cet outil vous aide à comprendre **pourquoi** le modèle a pris sa décision.")
    
    # Sélection du modèle
    col_model, _ = st.columns([1, 1])
    with col_model:
        model_choice = st.radio(
            "Modèle à analyser :",
            ["LSTM (recommandé)", "RNN"],
            horizontal=True
        )
    
    text_input = st.text_area(
        "✍️ Critique à analyser :",
        height=150,
        placeholder="Exemple: This movie was absolutely amazing! The acting was superb..."
    )
    
    if st.button("🔍 Interpréter", type="primary"):
        if text_input.strip():
            with st.spinner("Analyse en cours..."):
                # Sélection du modèle
                selected_model = lstm_model if "LSTM" in model_choice else rnn_model
                model_name = "LSTM" if "LSTM" in model_choice else "RNN"
                
                # Prédiction
                prob = predict_sentiment(text_input, selected_model, vocab, device)
                
                # Mots importants
                important_words = get_word_importance(text_input, selected_model, vocab, device)
                
                # Analyse de cohérence
                inconsistencies = check_text_coherence(text_input, prob, important_words)
                
                # Explication
                explanation = explain_prediction(text_input, prob, important_words, threshold)
                
                # Graphique d'importance
                fig = plot_word_importance(important_words, prob, f"Analyse {model_name}")
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown(explanation)
            
            if fig:
                st.pyplot(fig)
            
            if inconsistencies:
                st.markdown("---")
                st.markdown("### ⚠️ Incohérences détectées")
                for inc in inconsistencies:
                    st.warning(inc)
            
            # Résumé simple
            st.markdown("---")
            st.markdown("### 📝 Résumé")
            if prob > threshold:
                st.success(f"✅ **Cette critique est POSITIVE** avec une probabilité de {prob:.1%}")
            else:
                st.error(f"❌ **Cette critique est NÉGATIVE** avec une probabilité de {prob:.1%}")
            
            # Recommandation
            if prob > 0.7 or prob < 0.3:
                st.info("🎯 **Prédiction fiable** - Le modèle est très confiant.")
            else:
                st.info("🤔 **Prédiction incertaine** - La critique est probablement mitigée.")
        else:
            st.warning("Veuillez entrer une critique à analyser.")

# ==================== EXEMPLES PRÉ-DÉFINIS ====================
EXAMPLES = {
    "🎉 Critique positive - Film génial": "This movie was absolutely incredible! The acting was superb, the plot kept me guessing until the end, and the visuals were stunning. I highly recommend it to everyone!",
    "💔 Critique négative - Film décevant": "What a waste of time. The acting was terrible, the story made no sense, and the special effects looked like they were from 1990. I want my money back.",
    "🤔 Critique mitigée - Film moyen": "It was okay. Some parts were good, others were boring. The main actor did a decent job but the script was weak. Not great, not terrible.",
    "🍿 Blockbuster - Action": "Non-stop action from beginning to end! Explosions, car chases, and a hero you can root for. Exactly what I wanted from this type of movie.",
    "🎭 Drame psychologique": "A slow burn that really makes you think. The character development is excellent, though some might find the pacing too slow. A masterpiece of modern cinema.",
    "😏 Critique sarcastique": "Yeah right, this movie is 'amazing' if you love wasting 2 hours of your life. The acting was 'brilliant' like a wooden plank. Sure, best film ever... NOT!",
    "🌐 Critique non-anglaise": "Ce film est absolument magnifique! Les acteurs sont incroyables et l'histoire est captivante. Je le recommande vivement!"
}

# ==================== INTERFACE STREAMLIT ====================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

st.title("🎬 Analyse de sentiments - Critiques de films IMDb")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Informations")
    st.markdown("""
    **Modèles comparés :**
    - 🔴 **RNN** : Réseau récurrent simple
    - 🔵 **LSTM** : Réseau à mémoire longue
    
    **Dataset :** IMDb Reviews (50 000 critiques)
    
    **Performances (test set) :**
    - RNN : **73.24%** accuracy
    - LSTM : **84.30%** accuracy
    """)
    
    st.markdown("---")
    st.header("📈 Évolution des performances")
    st.markdown("""
    | Epoch | RNN Acc | LSTM Acc |
    |-------|---------|----------|
    | 1 | 57.28% | 64.20% |
    | 5 | 75.57% | 86.56% |
    | 10 | 86.50% | 95.04% |
    """)
    
    st.markdown("---")
    st.header("⚙️ Paramètres")
    threshold = st.slider("🎯 Seuil de classification", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    
    # Statistiques en temps réel
    st.header("📊 Statistiques en temps réel")
    total_analyses = len(st.session_state.history)
    if total_analyses > 0:
        agreements = sum(1 for h in st.session_state.history 
                        if (h['rnn_prob'] > 0.5) == (h['lstm_prob'] > 0.5))
        st.metric("Total analyses", total_analyses)
        st.metric("Accord RNN/LSTM", f"{agreements/total_analyses:.1%}")
    else:
        st.info("Aucune analyse encore effectuée")
    
    st.markdown("---")
    st.markdown("""
    **Auteurs :** SYLVAIN & MARTIN  
    **Date :** 12 Juin 2026
    """)

# Chargement des modèles
with st.spinner("Chargement des modèles..."):
    rnn_model, lstm_model, vocab, device, models_ok, errors = load_models()

if not models_ok:
    st.error(f"❌ Fichiers manquants : {', '.join(errors)}")
    st.info("Veuillez exécuter les notebooks d'entraînement d'abord pour générer les modèles.")
    st.stop()

# Onglets principaux (6 onglets maintenant)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Analyse simple", 
    "📊 Mode batch", 
    "🔬 Analyse avancée", 
    "🧠 Intelligence",
    "🔮 Interprétation",
    "📜 Historique & Tests"
])

# ==================== TAB 1: ANALYSE SIMPLE ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Entrez votre critique de film")
        
        example_choice = st.selectbox("📝 Ou choisissez un exemple :", ["(Personnalisé)"] + list(EXAMPLES.keys()), key="example_select")
        
        if example_choice != "(Personnalisé)":
            user_input = EXAMPLES[example_choice]
        else:
            user_input = ""
        
        user_input = st.text_area(
            "Critique à analyser :",
            value=user_input,
            height=200,
            placeholder="Exemple: This movie was absolutely amazing! The acting was superb..."
        )
        
        # Détection de langue
        if user_input.strip():
            lang = detect_language(user_input)
            if lang != 'en':
                st.warning(f"⚠️ Langue détectée : {lang.upper()}. Le modèle est optimisé pour l'anglais.")
        
        # Longueur du texte
        word_count = len(clean_text(user_input).split()) if user_input.strip() else 0
        if word_count > MAX_LEN:
            st.warning(f"⚠️ Texte tronqué : {word_count} > {MAX_LEN} mots")
        else:
            st.caption(f"📏 {word_count} mots")
        
        predict_button = st.button("🔍 Analyser le sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistiques")
        st.metric("📊 Taille vocabulaire", f"{VOCAB_SIZE:,} mots")
        st.metric("🔤 Longueur max", f"{MAX_LEN} mots")
        st.metric("🎯 Classes", "Positif / Négatif")
        st.metric("🎚️ Seuil", f"{threshold:.2f}")
    
    if predict_button and user_input.strip():
        if len(user_input.strip()) < 5:
            st.warning("⚠️ Veuillez entrer une critique plus longue (au moins 5 caractères).")
        else:
            with st.spinner("Analyse en cours..."):
                # Détection de sarcasme
                sarcasm_score = detect_sarcasm(user_input)
                if sarcasm_score > 0.3:
                    st.warning(f"⚠️ Sarcasme potentiel détecté ({sarcasm_score:.0%}). La prédiction peut être moins fiable.")
                
                # Prédictions avec incertitude
                prob_rnn_mean, prob_rnn_std = predict_with_uncertainty(user_input, rnn_model, vocab, device)
                prob_lstm_mean, prob_lstm_std = predict_with_uncertainty(user_input, lstm_model, vocab, device)
                
                sentiment_rnn = "POSITIF 😊" if prob_rnn_mean > threshold else "NÉGATIF 😞"
                sentiment_lstm = "POSITIF 😊" if prob_lstm_mean > threshold else "NÉGATIF 😞"
                
                # Historique
                st.session_state.history.append({
                    'text': user_input[:200],
                    'rnn_prob': prob_rnn_mean,
                    'lstm_prob': prob_lstm_mean,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'sarcasm': sarcasm_score
                })
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]
            
            # Résultats
            st.markdown("---")
            st.subheader("📊 Résultats de l'analyse")
            
            # Incertitude
            st.info(f"📊 Incertitude RNN : {prob_rnn_std:.2%} | Incertitude LSTM : {prob_lstm_std:.2%}")
            
            # Confiance combinée
            combined_confidence = (prob_rnn_mean + prob_lstm_mean) / 2
            if combined_confidence > 0.7:
                st.success(f"✅ **Confiance combinée élevée : {combined_confidence:.1%}**")
            elif combined_confidence < 0.3:
                st.error(f"❌ **Confiance combinée faible : {combined_confidence:.1%}**")
            else:
                st.info(f"ℹ️ **Confiance combinée moyenne : {combined_confidence:.1%}**")
            
            col_rnn, col_lstm = st.columns(2)
            
            with col_rnn:
                st.markdown("### 🔴 Modèle RNN")
                if prob_rnn_mean > threshold:
                    st.success(f"**Sentiment :** {sentiment_rnn}")
                else:
                    st.error(f"**Sentiment :** {sentiment_rnn}")
                st.metric("Probabilité positif", f"{prob_rnn_mean:.2%}")
                st.metric("Incertitude", f"{prob_rnn_std:.2%}", delta="±")
                st.progress(prob_rnn_mean)
            
            with col_lstm:
                st.markdown("### 🔵 Modèle LSTM")
                if prob_lstm_mean > threshold:
                    st.success(f"**Sentiment :** {sentiment_lstm}")
                else:
                    st.error(f"**Sentiment :** {sentiment_lstm}")
                st.metric("Probabilité positif", f"{prob_lstm_mean:.2%}")
                st.metric("Incertitude", f"{prob_lstm_std:.2%}", delta="±")
                st.progress(prob_lstm_mean)
            
            # Graphique
            fig, ax = plt.subplots(figsize=(10, 5))
            models = ['RNN', 'LSTM']
            probs = [prob_rnn_mean, prob_lstm_mean]
            errors = [prob_rnn_std, prob_lstm_std]
            colors = ['#e74c3c', '#3498db']
            bars = ax.bar(models, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2, yerr=errors, capsize=5)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probabilité (sentiment positif)', fontsize=12)
            ax.set_title('Comparaison RNN vs LSTM (avec incertitude)', fontsize=14, fontweight='bold')
            ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Seuil ({threshold:.2f})')
            ax.legend(loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob:.1%}', ha='center', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            
            # Suggestion d'amélioration
            st.markdown("---")
            st.subheader("💡 Suggestion d'amélioration")
            improvements = suggest_improvements(user_input, prob_lstm_mean)
            if improvements:
                for imp in improvements:
                    st.info(f"✏️ {imp}")
            else:
                st.success("✅ Aucune suggestion d'amélioration détectée.")
            
            # Feedback utilisateur
            st.markdown("---")
            st.subheader("📝 Votre avis")
            feedback = st.radio("Cette prédiction était-elle correcte ?", ["✅ Oui, correcte", "❌ Non, incorrecte"], key="feedback_radio", horizontal=True)
            if st.button("Envoyer le feedback", key="feedback_btn"):
                st.session_state.feedback_data.append({
                    'text': user_input[:100],
                    'rnn_prob': prob_rnn_mean,
                    'lstm_prob': prob_lstm_mean,
                    'feedback': feedback,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Merci pour votre feedback !")

# ==================== TAB 2: MODE BATCH ====================
with tab2:
    st.subheader("📊 Analyse par lots")
    batch_texts = st.text_area("Critiques (une par ligne) :", height=200)
    batch_model = st.radio("Modèle à utiliser :", ["LSTM (recommandé)", "RNN"], horizontal=True)
    
    if st.button("🔍 Analyser le batch", type="primary"):
        if batch_texts.strip():
            texts = [t.strip() for t in batch_texts.split('\n') if t.strip()]
            with st.spinner(f"Analyse de {len(texts)} critiques..."):
                model = lstm_model if "LSTM" in batch_model else rnn_model
                model_name = "LSTM" if "LSTM" in batch_model else "RNN"
                results = analyze_batch(texts, model, vocab, device, MAX_LEN)
            
            st.markdown("---")
            st.subheader(f"📊 Résultats du batch ({model_name})")
            
            pos_count = sum(1 for r in results if r["probability"] > threshold)
            neg_count = len(results) - pos_count
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("😊 Positifs", pos_count)
            with col_b2:
                st.metric("😞 Négatifs", neg_count)
            
            for res in results:
                if res["probability"] > threshold:
                    st.success(f"**{res['id']}. POSITIF** ({res['probability']:.1%}) - {res['text']}")
                else:
                    st.error(f"**{res['id']}. NÉGATIF** ({res['probability']:.1%}) - {res['text']}")

# ==================== TAB 3: ANALYSE AVANCÉE ====================
with tab3:
    st.subheader("🔬 Analyse avancée")
    adv_text = st.text_area("Critique à analyser en détail :", height=150)
    
    if adv_text.strip():
        # Mots influents
        st.markdown("#### 📝 Mots les plus influents")
        with st.spinner("Analyse..."):
            important_words = get_word_importance(adv_text, lstm_model, vocab, device, MAX_LEN)
            if important_words:
                for word, importance in important_words[:10]:
                    st.markdown(f"- **{word}** : impact {importance:.2%}")
            else:
                st.info("Aucun mot significatif détecté.")
        
        # Analyse phrase par phrase
        st.markdown("#### 📄 Analyse phrase par phrase")
        with st.spinner("Analyse des phrases..."):
            sentence_results = analyze_sentence_level(adv_text, lstm_model, vocab, device)
            if sentence_results:
                fig_s, ax_s = plt.subplots(figsize=(10, 4))
                sentences_idx = [r["index"] for r in sentence_results]
                sentences_prob = [r["probability"] for r in sentence_results]
                ax_s.bar(sentences_idx, sentences_prob, color='#3498db', alpha=0.7)
                ax_s.axhline(y=threshold, color='red', linestyle='--', label=f'Seuil ({threshold:.2f})')
                ax_s.set_xlabel('Phrase')
                ax_s.set_ylabel('Probabilité positif')
                ax_s.set_title('Évolution du sentiment par phrase')
                ax_s.legend()
                ax_s.set_ylim(0, 1)
                st.pyplot(fig_s)
            else:
                st.info("Texte trop court pour l'analyse phrase par phrase")
        
        # Longueur optimale
        st.markdown("#### 📏 Impact de la longueur")
        with st.spinner("Analyse..."):
            length_results = optimal_length_analysis(adv_text, lstm_model, vocab, device)
            if length_results:
                fig_l, ax_l = plt.subplots(figsize=(10, 4))
                lengths = [r["length"] for r in length_results]
                probs = [r["probability"] for r in length_results]
                ax_l.plot(lengths, probs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
                ax_l.axhline(y=threshold, color='red', linestyle='--', label=f'Seuil ({threshold:.2f})')
                ax_l.set_xlabel('Longueur (mots)')
                ax_l.set_ylabel('Probabilité positif')
                ax_l.set_title('Impact de la longueur sur la prédiction')
                ax_l.legend()
                ax_l.grid(True, alpha=0.3)
                st.pyplot(fig_l)
        
        # Sensibilité
        st.markdown("#### 🔧 Analyse de sensibilité")
        word_to_test = st.text_input("Mot à tester :", placeholder="ex: amazing, terrible, good")
        if word_to_test and st.button("Tester l'impact", key="sensitivity_btn"):
            base, new, diff = mask_word_importance(adv_text, word_to_test, lstm_model, vocab, device, MAX_LEN)
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Originale", f"{base:.2%}")
            with col_s2:
                st.metric("Après masquage", f"{new:.2%}")
            with col_s3:
                st.metric("Impact", f"{diff:.2%}", delta="±")
        
        # Word Cloud
        st.markdown("#### ☁️ Nuage de mots")
        if st.button("Générer le nuage de mots", key="wordcloud_btn"):
            try:
                from wordcloud import WordCloud
                text_clean = clean_text(adv_text)
                if text_clean:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_clean)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                else:
                    st.warning("Texte trop court")
            except ImportError:
                st.info("pip install wordcloud")

# ==================== TAB 4: INTELLIGENCE ====================
with tab4:
    st.subheader("🧠 Systèmes d'intelligence avancée")
    
    intel_text = st.text_area("Critique à analyser avec l'intelligence avancée :", height=150)
    
    if intel_text.strip():
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            # Détection de langue
            st.markdown("#### 🌐 Détection de langue")
            lang = detect_language(intel_text)
            st.metric("Langue détectée", lang.upper() if lang != 'unknown' else "Non détectée")
            if lang != 'en' and lang != 'unknown':
                st.warning("⚠️ Le modèle est optimisé pour l'anglais")
            
            # Détection de sarcasme
            st.markdown("#### 😏 Détection de sarcasme")
            sarcasm_score = detect_sarcasm(intel_text)
            st.progress(sarcasm_score)
            if sarcasm_score > 0.3:
                st.warning(f"Sarcasme potentiel : {sarcasm_score:.0%}")
            else:
                st.success(f"Sarcasme faible : {sarcasm_score:.0%}")
        
        with col_i2:
            # Incertitude
            st.markdown("#### 📊 Incertitude de prédiction")
            with st.spinner("Calcul de l'incertitude..."):
                _, prob_rnn_std = predict_with_uncertainty(intel_text, rnn_model, vocab, device)
                _, prob_lstm_std = predict_with_uncertainty(intel_text, lstm_model, vocab, device)
            st.metric("Incertitude RNN", f"{prob_rnn_std:.2%}")
            st.metric("Incertitude LSTM", f"{prob_lstm_std:.2%}")
            if prob_lstm_std > 0.15:
                st.warning("⚠️ Haute incertitude sur cette prédiction")
        
        # Suggestion d'amélioration
        st.markdown("#### ✏️ Suggestion d'amélioration")
        prob_lstm = predict_sentiment(intel_text, lstm_model, vocab, device)
        improvements = suggest_improvements(intel_text, prob_lstm)
        if improvements:
            for imp in improvements:
                st.info(f"✏️ {imp}")
        
        # Test A/B
        st.markdown("#### 🧪 Test A/B - RNN vs LSTM")
        if st.button("Lancer le test A/B", key="ab_test_btn"):
            prob_rnn = predict_sentiment(intel_text, rnn_model, vocab, device)
            prob_lstm = predict_sentiment(intel_text, lstm_model, vocab, device)
            
            st.session_state.test_results.append({
                'text': intel_text[:100],
                'rnn_prob': prob_rnn,
                'lstm_prob': prob_lstm,
                'winner': "LSTM" if abs(prob_lstm-0.5) > abs(prob_rnn-0.5) else "RNN",
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            col_ab1, col_ab2, col_ab3 = st.columns(3)
            with col_ab1:
                st.metric("RNN", f"{prob_rnn:.2%}")
            with col_ab2:
                st.metric("LSTM", f"{prob_lstm:.2%}")
            with col_ab3:
                winner = "LSTM" if abs(prob_lstm-0.5) > abs(prob_rnn-0.5) else "RNN"
                st.metric("Gagnant", winner)

# ==================== TAB 5: INTERPRÉTATION ====================
with tab5:
    display_interpretation_tab(lstm_model, vocab, device, threshold)

# ==================== TAB 6: HISTORIQUE & TESTS ====================
with tab6:
    st.subheader("📜 Historique des analyses")
    
    if st.session_state.history:
        st.markdown("**Dernières analyses :**")
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analyse {i+1} - {entry['timestamp']}"):
                st.markdown(f"**Texte :** {entry['text']}...")
                if 'sarcasm' in entry:
                    st.markdown(f"**Sarcasme détecté :** {entry['sarcasm']:.0%}")
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.metric("RNN", f"{entry['rnn_prob']:.2%}")
                with col_h2:
                    st.metric("LSTM", f"{entry['lstm_prob']:.2%}")
        
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Aucune analyse dans l'historique")
    
    # Feedback utilisateur
    st.markdown("---")
    st.subheader("📝 Feedback utilisateur")
    if st.session_state.feedback_data:
        st.markdown(f"**{len(st.session_state.feedback_data)}** retours enregistrés")
        if st.button("Afficher les feedbacks"):
            for fb in st.session_state.feedback_data[-5:]:
                st.markdown(f"- {fb['timestamp']} : {fb['feedback']} - {fb['text']}...")
    else:
        st.info("Aucun feedback enregistré")
    
    # Résultats des tests A/B
    st.markdown("---")
    st.subheader("🧪 Résultats des tests A/B")
    if st.session_state.test_results:
        st.markdown(f"**{len(st.session_state.test_results)}** tests effectués")
        lstm_wins = sum(1 for t in st.session_state.test_results if t['winner'] == "LSTM")
        rnn_wins = len(st.session_state.test_results) - lstm_wins
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric("LSTM gagnant", f"{lstm_wins}", delta=f"{lstm_wins/len(st.session_state.test_results):.0%}")
        with col_w2:
            st.metric("RNN gagnant", f"{rnn_wins}", delta=f"{rnn_wins/len(st.session_state.test_results):.0%}")
    else:
        st.info("Aucun test A/B effectué. Utilisez l'onglet 'Intelligence' pour tester.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>📊 Projet NLP - Comparaison RNN vs LSTM pour la classification de sentiments</p>
    <p>🎬 Dataset: IMDb Reviews (HuggingFace) | ⚙️ Modèles entraînés sur 10 epochs | 🏆 Meilleur modèle: LSTM (84.30% accuracy)</p>
    <p>🧠 Intelligence avancée : Détection de langue, Sarcasme, Incertitude, Test A/B, Feedback utilisateur, Interprétation intelligente</p>
</div>
""", unsafe_allow_html=True)