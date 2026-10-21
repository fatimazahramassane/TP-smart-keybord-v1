import streamlit as st
from src.pipeline import (
    NGramModel, InterpolatedModel, load_documents, split_by_document, 
    backoff_top_k, top_k_accuracy, EOS, UNK
)

st.set_page_config(page_title="Smart Keyboard V2", page_icon="", layout="wide")

st.title(" Smart Keyboard V2")
st.caption("Prédiction du mot suivant avec des modèles de langage N-Gram et lissage Add-k")

# --- Chargement en cache des données et modèles ---
@st.cache_resource
def load_models_and_data():
    docs = load_documents()
    train, test = split_by_document(docs)
    return docs, train, test

docs, train, test = load_models_and_data()

# --- Configuration dans la barre latérale ---
st.sidebar.header(" Configuration du Modèle")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle de prédiction",
    ["trigram", "bigram", "unigram", "interpolated", "backoff"]
)

k_value = st.sidebar.slider("Constante de lissage (k)", 0.0, 1.0, 0.1, 0.01)

# --- Fonction pour obtenir le modèle actif ---
def get_active_model(choice, k):
    if choice == "backoff":
        return {
            "tri": NGramModel(train, 3, k),
            "bi": NGramModel(train, 2, k),
            "uni": NGramModel(train, 1, k)
        }
    elif choice == "interpolated":
        return InterpolatedModel(train, k=k, lambdas=(0.6, 0.3, 0.1))
    else:
        n = 3 if choice == "trigram" else 2 if choice == "bigram" else 1
        return NGramModel(train, n, k)

active_model = get_active_model(model_choice, k_value)

# --- Interface Principale ---
st.markdown("###  Composition de texte")
st.markdown("Commencez à taper ou cliquez sur les suggestions pour construire votre phrase.")

# Gestion de l'état du texte
if "current_text" not in st.session_state:
    st.session_state.current_text = "the happy girls"

def add_word(new_word):
    current = st.session_state.current_text.strip()
    if new_word == EOS:
        st.session_state.current_text = current + "."
    else:
        st.session_state.current_text = current + " " + new_word

# Zone de saisie liée à l'état de session
user_input = st.text_input("Phrase en cours :", key="current_text")
words = user_input.lower().replace(".", " . ").split()

# --- Affichage des Suggestions ---
st.markdown("###  Suggestions (Top-3)")

if not words:
    st.info("Tapez au moins un mot pour obtenir des suggestions.")
else:
    # Vérification des mots inconnus
    vocab = active_model.vocab if hasattr(active_model, 'vocab') else active_model['uni'].vocab
    unknown = [w for w in words if w not in vocab]
    if unknown:
        st.warning(f" Mots hors vocabulaire (traités comme <unk>) : {', '.join(set(unknown))}")

    # Récupération des suggestions
    if model_choice == "backoff":
        suggestions, order = backoff_top_k(active_model["tri"], active_model["bi"], active_model["uni"], words, 3)
        st.caption(f"Mécanisme : Back-off dur (a utilisé le modèle {order})")
    else:
        suggestions = active_model.top_k(words, 3)
        order = model_choice
        st.caption(f"Mécanisme : Modèle {order} direct")

    cols = st.columns(3)
    for i, (col, (w, p)) in enumerate(zip(cols, suggestions)):
        with col:
            label = " (Fin de phrase)" if w == EOS else w
            st.metric(label=f"Suggestion {i+1}", value=label)
            st.progress(min(p, 1.0), text=f"Probabilité : {p:.1%}")
            
            # Bouton unique pour éviter les conflits de clés
            btn_key = f"add_{i}_{w}_{hash(user_input)}"
           #if st.button(f"➕ Ajouter", key=btn_key):
             #   add_word(w)
            #    st.rerun()

# --- Onglet d'Évaluation ---
st.markdown("---")
with st.expander(" Voir les métriques d'évaluation (Jeu de test)", expanded=False):
    st.markdown("Performance du modèle sur les documents de test (20% des données).")
    
    # Pour l'affichage, on évalue le modèle trigramme de base avec le k choisi
    eval_model = NGramModel(train, 3, k_value)
    ppl, zero_prob = eval_model.perplexity(test)
    acc, _ = top_k_accuracy(eval_model, test, 3)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Perplexité", f"{ppl:.2f}" if ppl != float("inf") else "∞", delta="Plus bas = mieux")
    col2.metric("Précision Top-3", f"{acc:.1%}", delta="Plus haut = mieux")
    col3.metric("Tokens à probabilité nulle", f"{zero_prob:.1%}", delta="Plus bas = mieux", delta_color="inverse")
    
    st.caption("Les métriques sont calculées dynamiquement en fonction de la valeur de k sélectionnée.")

# --- Pied de page ---
st.markdown("---")
st.markdown("Développé avec  via Streamlit | [Dépôt GitHub](https://github.com/fatimazahramassane/TP-smart-keybord-v1)")