# 🚀 ANALIZZATORE TRUSTPILOT - STREAMLIT APP
# Versione Web User-Friendly

import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches
import warnings
import io
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="🚀 Analizzatore Trustpilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 CSS PERSONALIZZATO
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 HEADER PRINCIPALE
st.markdown('<h1 class="main-header">🚀 Analizzatore Trustpilot Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Cosa fa questa App?</h3>
    <p>• Estrae automaticamente recensioni da Trustpilot</p>
    <p>• Analizza sentiment e identifica punti di forza/debolezza</p>
    <p>• Genera strategie di Digital Marketing personalizzate</p>
    <p>• Fornisce suggerimenti specifici per SEO, ADV, Email e CRO</p>
</div>
""", unsafe_allow_html=True)

# 🔧 FUNZIONI BACKEND (nascoste nell'interfaccia)
@st.cache_data
def get_stopwords():
    return set([
        "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per", 
        "tra", "fra", "un", "una", "uno", "e", "ma", "anche", "come", "che", "non", 
        "più", "meno", "molto", "poco", "tutto", "tutti", "tutte", "questo", "questa", 
        "questi", "queste", "quello", "quella", "quelli", "quelle", "sono", "è", "ho", 
        "hai", "ha", "hanno", "essere", "avere", "fare", "dire", "andare", "del", "della",
        "dei", "delle", "dal", "dalla", "dai", "dalle", "nel", "nella", "nei", "nelle",
        "sul", "sulla", "sui", "sulle", "al", "alla", "ai", "alle", "ho", "ottimo",
        "buono", "buona", "bene", "male", "servizio", "prodotto", "azienda", "sempre"
    ])

def pulisci_testo(testo):
    """Pulizia avanzata del testo"""
    stopwords_italiane = get_stopwords()
    testo = testo.lower()
    testo = re.sub(r'pubblicata il \d{1,2} \w+ \d{4}', '', testo)
    testo = re.sub(r'stelle su 5', '', testo)
    testo = re.sub(r'\d+ stelle', '', testo)
    testo = re.sub(r'data dell\'esperienza:.*', '', testo)
    testo = re.sub(r'[^\w\s]', ' ', testo)
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    
    parole = testo.split()
    parole_filtrate = [parola for parola in parole if parola not in stopwords_italiane and len(parola) > 2]
    return " ".join(parole_filtrate)

def estrai_recensioni(url_base, max_pagine, progress_bar, status_text):
    """Estrae recensioni con barra di progresso"""
    tutte_recensioni = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for pagina in range(1, max_pagine + 1):
        url_pagina = f"{url_base}?page={pagina}"
        status_text.text(f"🕷️ Estrazione pagina {pagina}/{max_pagine}...")
        
        try:
            response = requests.get(url_pagina, headers=headers, timeout=10)
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            
            selettori = [
                'section[class*="reviewContentwrapper"]',
                'div[class*="review-content"]',
                'article[class*="review"]'
            ]
            
            sezioni = []
            for selettore in selettori:
                sezioni.extend(soup.select(selettore))
            
            if not sezioni:
                if pagina == 1:
                    st.error("❌ Nessuna recensione trovata. Verifica l'URL.")
                    return []
                continue

            for sezione in sezioni:
                testo = sezione.get_text(separator=" ", strip=True)
                if testo and len(testo) > 50:
                    tutte_recensioni.append(testo)
            
            progress_bar.progress(pagina / max_pagine)
            time.sleep(1)

        except Exception as e:
            st.warning(f"⚠️ Errore pagina {pagina}: {e}")
            continue

    return tutte_recensioni

def deduplica_avanzata(lista, soglia=0.8):
    """Deduplica concetti simili"""
    if not lista:
        return []
    
    risultati = []
    lista_copia = lista.copy()
    
    while lista_copia:
        elemento = lista_copia.pop(0)
        simili = get_close_matches(elemento, lista_copia, cutoff=soglia)
        
        for simile in simili:
            if simile in lista_copia:
                lista_copia.remove(simile)
        
        risultati.append(elemento)
    
    return risultati

def analizza_blocchi_avanzata(blocchi, client, progress_bar, status_text):
    """Analisi AI con barra di progresso"""
    risultati = {
        "punti_forza": [],
        "punti_debolezza": [],
        "leve_marketing": [],
        "parole_chiave": [],
        "suggerimenti_seo": [],
        "suggerimenti_adv": [],
        "suggerimenti_email": [],
        "suggerimenti_cro": [],
        "suggerimenti_sinergie": []
    }

    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 Analizzando blocco {i+1}/{len(blocchi)} con AI...")

        prompt = f"""
        Analizza le seguenti recensioni di un'azienda e fornisci insights strategici:

        RECENSIONI:
        {blocco}

        Rispondi SOLO in formato JSON valido con queste chiavi:
        {{
            "punti_forza": ["punto1", "punto2", ...],
            "punti_debolezza": ["problema1", "problema2", ...],
            "leve_marketing": ["leva1", "leva2", ...],
            "parole_chiave": ["parola1", "parola2", ...],
            "suggerimenti_seo": ["suggerimento SEO specifico 1", "suggerimento SEO specifico 2", ...],
            "suggerimenti_adv": ["strategia pubblicitaria 1", "strategia pubblicitaria 2", ...],
            "suggerimenti_email": ["strategia email 1", "strategia email 2", ...],
            "suggerimenti_cro": ["ottimizzazione conversioni 1", "ottimizzazione conversioni 2", ...],
            "suggerimenti_sinergie": ["sinergia tra canali 1", "sinergia tra canali 2", ...]
        }}

        LINEE GUIDA:
        - Punti di forza: aspetti positivi concreti e specifici
        - Punti di debolezza: problemi ricorrenti identificabili
        - Leve marketing: vantaggi competitivi sfruttabili
        - Parole chiave: termini rilevanti per il business (no stopwords)
        - SEO: suggerimenti per contenuti, pagine, parole chiave specifiche
        - ADV: strategie pubblicitarie concrete con creatività e targeting
        - Email: automazioni, segmentazioni, contenuti email specifici
        - CRO: ottimizzazioni UX/UI concrete per migliorare conversioni
        - Sinergie: come integrare i canali digital per massimizzare risultati

        Fornisci suggerimenti PRATICI e IMPLEMENTABILI, non generici.
        """

        for tentativo in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )

                content = response.choices[0].message.content
                content_cleaned = re.sub(r"```json\n?|```", "", content).strip()
                
                dati = json.loads(content_cleaned)

                for chiave in risultati:
                    if chiave in dati:
                        nuovi_elementi = [elem for elem in dati[chiave] if elem not in risultati[chiave]]
                        risultati[chiave].extend(nuovi_elementi)

                break
                
            except json.JSONDecodeError as e:
                if tentativo < 2:
                    time.sleep(2)
            except Exception as e:
                if tentativo < 2:
                    time.sleep(5)

        progress_bar.progress((i + 1) / len(blocchi))

    # Deduplica
    for chiave in risultati:
        risultati[chiave] = deduplica_avanzata(list(set(risultati[chiave])), soglia=0.75)
    
    return risultati

def crea_excel_download(recensioni_raw, recensioni_pulite, risultati):
    """Crea file Excel per download"""
    output = io.BytesIO()
    
    df_recensioni = pd.DataFrame({
        "Recensioni Originali": recensioni_raw,
        "Recensioni Pulite": recensioni_pulite
    })
    
    df_analisi = pd.DataFrame({
        "Categoria": ["Punti di Forza", "Punti di Debolezza", "Leve Marketing", "Parole Chiave"],
        "Insights": [
            " | ".join(risultati["punti_forza"][:10]),
            " | ".join(risultati["punti_debolezza"][:10]),
            " | ".join(risultati["leve_marketing"][:10]),
            " | ".join(risultati["parole_chiave"][:20])
        ]
    })
    
    df_digital = pd.DataFrame({
        "Canale": ["SEO", "ADV", "Email Marketing", "CRO", "Sinergie"],
        "Suggerimenti Specifici": [
            " | ".join(risultati["suggerimenti_seo"]),
            " | ".join(risultati["suggerimenti_adv"]),
            " | ".join(risultati["suggerimenti_email"]),
            " | ".join(risultati["suggerimenti_cro"]),
            " | ".join(risultati["suggerimenti_sinergie"])
        ]
    })
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_recensioni.to_excel(writer, sheet_name="Recensioni", index=False)
        df_analisi.to_excel(writer, sheet_name="Analisi Generale", index=False)
        df_digital.to_excel(writer, sheet_name="Strategia Digital", index=False)
    
    return output.getvalue()

# 🎮 INTERFACCIA PRINCIPALE
def main():
    # SIDEBAR PER INPUT
    with st.sidebar:
        st.markdown("## 🔧 Configurazione")
        
        # API Key
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Inserisci la tua API Key di OpenAI"
        )
        
        # URL Trustpilot
        url_base = st.text_input(
            "🌐 URL Trustpilot",
            placeholder="https://it.trustpilot.com/review/nomesito.com",
            help="Inserisci l'URL completo della pagina Trustpilot"
        )
        
        # Numero pagine
        max_pagine = st.slider(
            "📄 Numero massimo pagine",
            min_value=1,
            max_value=50,
            value=10,
            help="Più pagine = più recensioni ma più tempo"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Suggerimenti")
        st.info("• Inizia con 5-10 pagine per test rapidi\n• Usa 20+ pagine per analisi complete\n• Ogni pagina ha ~20 recensioni")

    # AREA PRINCIPALE
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Inizia l'Analisi")
        
        if st.button("🔍 Avvia Analisi Trustpilot", type="primary"):
            # Validazione input
            if not api_key:
                st.error("❌ Inserisci la tua OpenAI API Key nella sidebar")
                return
            
            if not url_base:
                st.error("❌ Inserisci l'URL di Trustpilot nella sidebar")
                return
            
            if "trustpilot.com" not in url_base:
                st.error("❌ L'URL deve essere di Trustpilot")
                return
            
            # Inizializza client OpenAI
            try:
                client = OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"❌ Errore API Key: {e}")
                return
            
            # Container per risultati
            results_container = st.container()
            
            with results_container:
                # FASE 1: Estrazione
                st.markdown("### 🕷️ Fase 1: Estrazione Recensioni")
                progress_bar_1 = st.progress(0)
                status_text_1 = st.empty()
                
                with st.spinner("Estrazione in corso..."):
                    recensioni_raw = estrai_recensioni(url_base, max_pagine, progress_bar_1, status_text_1)
                
                if not recensioni_raw:
                    st.error("❌ Nessuna recensione estratta. Verifica l'URL.")
                    return
                
                st.success(f"✅ Estratte {len(recensioni_raw)} recensioni!")
                
                # FASE 2: Pulizia
                st.markdown("### 🧹 Fase 2: Pulizia Dati")
                with st.spinner("Pulizia testi in corso..."):
                    recensioni_pulite = [pulisci_testo(r) for r in recensioni_raw]
                st.success("✅ Pulizia completata!")
                
                # FASE 3: Preparazione
                st.markdown("### 📝 Fase 3: Preparazione per AI")
                testo_completo = " ".join(recensioni_pulite)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"📊 Creati {len(blocchi)} blocchi per l'analisi AI")
                
                # FASE 4: Analisi AI
                st.markdown("### 🤖 Fase 4: Analisi AI")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                with st.spinner("Analisi AI in corso..."):
                    risultati = analizza_blocchi_avanzata(blocchi, client, progress_bar_2, status_text_2)
                
                st.markdown('<div class="success-box"><h3>🎉 Analisi Completata!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI
                st.markdown("## 📊 Risultati Analisi")
                
                # Metriche
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("📝 Recensioni", len(recensioni_raw))
                
                with col_m2:
                    st.metric("💪 Punti Forza", len(risultati['punti_forza']))
                
                with col_m3:
                    st.metric("⚠️ Criticità", len(risultati['punti_debolezza']))
                
                with col_m4:
                    st.metric("🎯 Leve Marketing", len(risultati['leve_marketing']))
                
                # Tabs per risultati
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["💪 Punti Forza", "⚠️ Criticità", "🎯 Leve Marketing", "📈 Strategie Digital", "🔍 Parole Chiave"])
                
                with tab1:
                    st.markdown("### 💪 Punti di Forza Identificati")
                    for i, punto in enumerate(risultati['punti_forza'][:10], 1):
                        st.markdown(f"**{i}.** {punto}")
                
                with tab2:
                    st.markdown("### ⚠️ Punti di Debolezza")
                    for i, punto in enumerate(risultati['punti_debolezza'][:10], 1):
                        st.markdown(f"**{i}.** {punto}")
                
                with tab3:
                    st.markdown("### 🎯 Leve Marketing")
                    for i, leva in enumerate(risultati['leve_marketing'][:10], 1):
                        st.markdown(f"**{i}.** {leva}")
                
                with tab4:
                    col_seo, col_adv = st.columns(2)
                    
                    with col_seo:
                        st.markdown("#### 🌐 SEO")
                        for sug in risultati['suggerimenti_seo'][:5]:
                            st.markdown(f"• {sug}")
                        
                        st.markdown("#### 📧 Email Marketing")
                        for sug in risultati['suggerimenti_email'][:5]:
                            st.markdown(f"• {sug}")
                    
                    with col_adv:
                        st.markdown("#### 📢 ADV")
                        for sug in risultati['suggerimenti_adv'][:5]:
                            st.markdown(f"• {sug}")
                        
                        st.markdown("#### 🔄 CRO")
                        for sug in risultati['suggerimenti_cro'][:5]:
                            st.markdown(f"• {sug}")
                    
                    st.markdown("#### 🤝 Sinergie tra Canali")
                    for sug in risultati['suggerimenti_sinergie'][:5]:
                        st.markdown(f"• {sug}")
                
                with tab5:
                    st.markdown("### 🔍 Parole Chiave Principali")
                    parole_cols = st.columns(3)
                    for i, parola in enumerate(risultati['parole_chiave'][:15]):
                        with parole_cols[i % 3]:
                            st.markdown(f"🔸 **{parola}**")
                
                # DOWNLOAD
                st.markdown("## 📥 Download Report")
                
                excel_data = crea_excel_download(recensioni_raw, recensioni_pulite, risultati)
                
                st.download_button(
                    label="📊 Scarica Report Excel Completo",
                    data=excel_data,
                    file_name="Analisi_Trustpilot_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
    
    with col2:
        st.markdown("## 📋 Guida Rapida")
        
        st.markdown("""
        ### 🎯 Come Usare:
        1. **Inserisci API Key** OpenAI nella sidebar
        2. **Copia URL** della pagina Trustpilot
        3. **Scegli numero** di pagine da analizzare
        4. **Clicca** "Avvia Analisi"
        5. **Scarica** il report Excel
        
        ### 🔑 Dove trovare API Key:
        1. Vai su [OpenAI](https://platform.openai.com)
        2. Accedi al tuo account
        3. Vai in "API Keys"
        4. Crea una nuova chiave
        
        ### 📈 Cosa Ottieni:
        • **Punti di forza** della tua azienda
        • **Criticità** da migliorare
        • **Strategie SEO** personalizzate
        • **Campagne ADV** suggerite
        • **Email Marketing** ottimizzato
        • **CRO** per conversioni migliori
        """)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚡ Prestazioni</h4>
            <p>L'analisi richiede 2-5 minuti per 10 pagine di recensioni</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()