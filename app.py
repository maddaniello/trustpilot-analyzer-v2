# 🚀 ANALIZZATORE TRUSTPILOT - STREAMLIT APP
# Versione Web User-Friendly con Clustering e Analisi Frequenze

import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from difflib import get_close_matches
import warnings
import io
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="🚀 Analizzatore Trustpilot Pro",
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
    
    .cluster-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .review-example {
        background: #e9ecef;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-style: italic;
    }
    
    .frequency-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 HEADER PRINCIPALE
st.markdown('<h1 class="main-header">🚀 Analizzatore Trustpilot Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Cosa fa questa App?</h3>
    <p>• Estrae automaticamente recensioni da Trustpilot</p>
    <p>• Clusterizza le recensioni per tematiche</p>
    <p>• Calcola la frequenza di ogni punto di forza/debolezza</p>
    <p>• Mostra esempi di recensioni per ogni tema</p>
    <p>• Genera strategie di Digital Marketing personalizzate</p>
</div>
""", unsafe_allow_html=True)

# 🔧 FUNZIONI BACKEND
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

def estrai_recensioni_con_metadata(url_base, max_pagine, progress_bar, status_text):
    """Estrae recensioni con metadata (URL specifico e rating)"""
    tutte_recensioni = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for pagina in range(1, max_pagine + 1):
        url_pagina = f"{url_base}?page={pagina}"
        status_text.text(f"🕷️ Estrazione pagina {pagina}/{max_pagine}...")
        
        try:
            response = requests.get(url_pagina, headers=headers, timeout=10)
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Approccio multi-selettore per maggiore compatibilità
            recensioni_elementi = []
            
            # Prova diversi selettori
            selettori = [
                'article[data-service-review-business-unit-id]',
                'div[class*="review-card"]',
                'div[class*="styles_reviewCard"]',
                'div[class*="paper_paper"]',
                'section[class*="reviewContentwrapper"]',
                'article.review',
                'div.review-content'
            ]
            
            for selettore in selettori:
                elementi = soup.select(selettore)
                if elementi:
                    recensioni_elementi.extend(elementi)
                    break
            
            # Se ancora vuoto, cerca in modo più generico
            if not recensioni_elementi:
                recensioni_elementi = soup.find_all(['article', 'div'], class_=re.compile(r'review|card'))
            
            for elemento in recensioni_elementi:
                try:
                    # Estrai testo
                    testo_completo = elemento.get_text(separator=" ", strip=True)
                    
                    # Estrai rating con metodi multipli
                    rating = 0
                    
                    # Metodo 1: Cerca attributi data
                    for attr in ['data-star-rating', 'data-rating', 'data-stars']:
                        if elemento.get(attr):
                            try:
                                rating = int(elemento.get(attr))
                                break
                            except:
                                pass
                    
                    # Metodo 2: Cerca nelle immagini stelle
                    if not rating:
                        star_imgs = elemento.find_all('img', alt=re.compile(r'\d\s*(star|stelle|su\s*5)', re.I))
                        if star_imgs:
                            for img in star_imgs:
                                match = re.search(r'(\d)', img.get('alt', ''))
                                if match:
                                    rating = int(match.group(1))
                                    break
                    
                    # Metodo 3: Conta elementi stella
                    if not rating:
                        # Cerca svg o span con classe star
                        stars = elemento.find_all(['svg', 'span', 'i'], class_=re.compile(r'star.*?(full|filled|active)|icon.*?star.*?full'))
                        if stars:
                            rating = len(stars)
                    
                    # Metodo 4: Estrai dal testo
                    if not rating:
                        rating_patterns = [
                            r'(\d)\s*stelle\s*su\s*5',
                            r'(\d)\s*su\s*5\s*stelle',
                            r'Valutazione:\s*(\d)',
                            r'Rating:\s*(\d)',
                            r'(\d)\s*star[s]?\s*out\s*of\s*5'
                        ]
                        for pattern in rating_patterns:
                            match = re.search(pattern, testo_completo, re.I)
                            if match:
                                rating = int(match.group(1))
                                break
                    
                    # Pulisci il testo principale della recensione
                    # Cerca il contenuto specifico della recensione
                    contenuto_recensione = None
                    
                    # Cerca in elementi specifici
                    for classe in ['reviewContent', 'review-content', 'review-text', 'typography_body']:
                        elem = elemento.find(class_=re.compile(classe, re.I))
                        if elem:
                            contenuto_recensione = elem.get_text(strip=True)
                            break
                    
                    # Se non trovato, cerca in p o div
                    if not contenuto_recensione:
                        p_elem = elemento.find('p')
                        if p_elem:
                            contenuto_recensione = p_elem.get_text(strip=True)
                        else:
                            contenuto_recensione = testo_completo
                    
                    # Pulisci il testo
                    testo = re.sub(r'Verificata|Data dell.*esperienza.*\d{4}|Pubblicata.*\d{4}', '', contenuto_recensione).strip()
                    
                    # Costruisci URL
                    review_url = url_pagina
                    link_elem = elemento.find('a', href=re.compile(r'/review'))
                    if link_elem:
                        href = link_elem['href']
                        if href.startswith('/'):
                            review_url = f"https://it.trustpilot.com{href}"
                        else:
                            review_url = href
                    
                    # Validazione base
                    if testo and len(testo) > 20:
                        # Se non abbiamo trovato il rating, assegnane uno di default
                        if rating == 0:
                            # Prova a dedurlo dal sentiment del testo
                            testo_lower = testo.lower()
                            if any(word in testo_lower for word in ['ottimo', 'eccellente', 'perfetto', 'consiglio']):
                                rating = 5
                            elif any(word in testo_lower for word in ['buono', 'bene', 'soddisfatto']):
                                rating = 4
                            elif any(word in testo_lower for word in ['pessimo', 'terribile', 'sconsiglio']):
                                rating = 1
                            else:
                                rating = 3  # Default neutro
                        
                        tutte_recensioni.append({
                            'testo': testo,
                            'testo_pulito': pulisci_testo(testo),
                            'url': review_url,
                            'pagina': pagina,
                            'rating': rating
                        })
                        
                except Exception as e:
                    continue
            
            # Se non abbiamo trovato recensioni e siamo alla prima pagina, mostra errore
            if not tutte_recensioni and pagina == 1:
                st.error(f"Debug: Nessuna recensione trovata nella pagina. HTML snippet: {str(soup.find('body'))[:200]}")
            
            progress_bar.progress(pagina / max_pagine)
            time.sleep(1)

        except Exception as e:
            st.warning(f"⚠️ Errore pagina {pagina}: {e}")
            continue

    return tutte_recensioni

def clusterizza_recensioni(recensioni_data, n_clusters=5):
    """Clusterizza le recensioni per tematiche"""
    if len(recensioni_data) < n_clusters:
        n_clusters = max(2, len(recensioni_data) // 2)
    
    # Estrai solo i testi puliti
    testi = [r['testo_pulito'] for r in recensioni_data]
    
    # Vettorizzazione TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(testi)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Aggiungi cluster ai dati
    for i, rec in enumerate(recensioni_data):
        rec['cluster'] = clusters[i]
    
    # Trova termini chiave per ogni cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    
    for i in range(n_clusters):
        # Trova il centroide del cluster
        cluster_center = kmeans.cluster_centers_[i]
        # Trova le parole più importanti
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        cluster_keywords[i] = top_keywords[:5]  # Top 5 keywords
    
    return recensioni_data, cluster_keywords

def analizza_frequenze_temi_migliorata(risultati, recensioni_data, client):
    """Calcola le frequenze usando l'AI per matching accurato"""
    analisi_frequenze = {
        'punti_forza': {},
        'punti_debolezza': {}
    }
    
    # Prepara recensioni per rating
    recensioni_positive = [r for r in recensioni_data if r['rating'] >= 4]
    recensioni_negative = [r for r in recensioni_data if r['rating'] <= 2]
    recensioni_medie = [r for r in recensioni_data if r['rating'] == 3]
    
    # Analizza punti di forza con AI
    if risultati['punti_forza'] and recensioni_positive:
        # Crea batch di recensioni positive
        batch_size = 10
        for punto in risultati['punti_forza'][:15]:  # Limita a 15 punti principali
            esempi_trovati = []
            count = 0
            
            # Analizza in batch per efficienza
            for i in range(0, min(len(recensioni_positive), 30), batch_size):
                batch = recensioni_positive[i:i+batch_size]
                recensioni_batch = "\n\n".join([f"[{j+1}] {r['testo'][:300]}" for j, r in enumerate(batch)])
                
                prompt = f"""
                Analizza se le seguenti recensioni POSITIVE menzionano questo punto di forza:
                "{punto}"
                
                RECENSIONI:
                {recensioni_batch}
                
                Rispondi SOLO con un JSON nel formato:
                {{
                    "recensioni_rilevanti": [1, 3, 5]  // numeri delle recensioni che menzionano VERAMENTE questo punto
                }}
                
                Includi SOLO recensioni che effettivamente parlano di questo aspetto positivo.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    content = response.choices[0].message.content
                    content_cleaned = re.sub(r"```json\n?|```", "", content).strip()
                    risultato = json.loads(content_cleaned)
                    
                    for idx in risultato.get('recensioni_rilevanti', []):
                        if 0 < idx <= len(batch):
                            rec = batch[idx-1]
                            count += 1
                            if len(esempi_trovati) < 2:
                                esempi_trovati.append({
                                    'testo': rec['testo'][:200] + '...' if len(rec['testo']) > 200 else rec['testo'],
                                    'url': rec['url'],
                                    'rating': rec['rating']
                                })
                
                except Exception:
                    continue
                
                if len(esempi_trovati) >= 2:
                    break
            
            if count > 0:
                percentuale = (count / len(recensioni_data)) * 100
                analisi_frequenze['punti_forza'][punto] = {
                    'count': count,
                    'percentuale': percentuale,
                    'esempi': esempi_trovati
                }
    
    # Analizza punti di debolezza con AI
    if risultati['punti_debolezza'] and recensioni_negative:
        batch_size = 10
        for punto in risultati['punti_debolezza'][:15]:
            esempi_trovati = []
            count = 0
            
            # Combina recensioni negative e medie
            recensioni_da_analizzare = recensioni_negative + recensioni_medie[:5]
            
            for i in range(0, min(len(recensioni_da_analizzare), 30), batch_size):
                batch = recensioni_da_analizzare[i:i+batch_size]
                recensioni_batch = "\n\n".join([f"[{j+1}] {r['testo'][:300]}" for j, r in enumerate(batch)])
                
                prompt = f"""
                Analizza se le seguenti recensioni NEGATIVE/CRITICHE menzionano questo problema:
                "{punto}"
                
                RECENSIONI:
                {recensioni_batch}
                
                Rispondi SOLO con un JSON nel formato:
                {{
                    "recensioni_rilevanti": [1, 3, 5]  // numeri delle recensioni che menzionano VERAMENTE questo problema
                }}
                
                Includi SOLO recensioni che effettivamente criticano questo aspetto.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    content = response.choices[0].message.content
                    content_cleaned = re.sub(r"```json\n?|```", "", content).strip()
                    risultato = json.loads(content_cleaned)
                    
                    for idx in risultato.get('recensioni_rilevanti', []):
                        if 0 < idx <= len(batch):
                            rec = batch[idx-1]
                            count += 1
                            if len(esempi_trovati) < 2:
                                esempi_trovati.append({
                                    'testo': rec['testo'][:200] + '...' if len(rec['testo']) > 200 else rec['testo'],
                                    'url': rec['url'],
                                    'rating': rec['rating']
                                })
                
                except Exception:
                    continue
                
                if len(esempi_trovati) >= 2:
                    break
            
            if count > 0:
                percentuale = (count / len(recensioni_data)) * 100
                analisi_frequenze['punti_debolezza'][punto] = {
                    'count': count,
                    'percentuale': percentuale,
                    'esempi': esempi_trovati
                }
    
    # Ordina per frequenza
    analisi_frequenze['punti_forza'] = dict(sorted(
        analisi_frequenze['punti_forza'].items(), 
        key=lambda x: x[1]['percentuale'], 
        reverse=True
    ))
    
    analisi_frequenze['punti_debolezza'] = dict(sorted(
        analisi_frequenze['punti_debolezza'].items(), 
        key=lambda x: x[1]['percentuale'], 
        reverse=True
    ))
    
    return analisi_frequenze

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

def crea_excel_download_avanzato(recensioni_data, risultati, analisi_frequenze, cluster_keywords):
    """Crea file Excel avanzato per download"""
    output = io.BytesIO()
    
    # DataFrame recensioni con cluster
    df_recensioni = pd.DataFrame({
        "Recensioni Originali": [r['testo'] for r in recensioni_data],
        "Recensioni Pulite": [r['testo_pulito'] for r in recensioni_data],
        "Cluster": [r['cluster'] for r in recensioni_data],
        "URL": [r['url'] for r in recensioni_data]
    })
    
    # DataFrame analisi con frequenze
    punti_forza_data = []
    for punto, dati in analisi_frequenze['punti_forza'].items():
        punti_forza_data.append({
            'Punto di Forza': punto,
            'Frequenza %': f"{dati['percentuale']:.1f}%",
            'Occorrenze': dati['count'],
            'Esempio 1': dati['esempi'][0]['testo'] if dati['esempi'] else '',
            'URL Esempio 1': dati['esempi'][0]['url'] if dati['esempi'] else ''
        })
    df_punti_forza = pd.DataFrame(punti_forza_data)
    
    punti_debolezza_data = []
    for punto, dati in analisi_frequenze['punti_debolezza'].items():
        punti_debolezza_data.append({
            'Punto di Debolezza': punto,
            'Frequenza %': f"{dati['percentuale']:.1f}%",
            'Occorrenze': dati['count'],
            'Esempio 1': dati['esempi'][0]['testo'] if dati['esempi'] else '',
            'URL Esempio 1': dati['esempi'][0]['url'] if dati['esempi'] else ''
        })
    df_punti_debolezza = pd.DataFrame(punti_debolezza_data)
    
    # DataFrame cluster
    cluster_data = []
    for cluster_id, keywords in cluster_keywords.items():
        cluster_data.append({
            'Cluster': f"Tema {cluster_id + 1}",
            'Parole Chiave': ' | '.join(keywords),
            'Num. Recensioni': sum(1 for r in recensioni_data if r['cluster'] == cluster_id)
        })
    df_clusters = pd.DataFrame(cluster_data)
    
    # DataFrame strategie digital
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
        df_clusters.to_excel(writer, sheet_name="Cluster Tematici", index=False)
        df_punti_forza.to_excel(writer, sheet_name="Punti Forza Frequenze", index=False)
        df_punti_debolezza.to_excel(writer, sheet_name="Punti Debolezza Frequenze", index=False)
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
        
        # Numero cluster
        n_clusters = st.slider(
            "🎯 Numero di temi (cluster)",
            min_value=3,
            max_value=10,
            value=5,
            help="Numero di tematiche principali da identificare"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Suggerimenti")
        st.info("• Inizia con 5-10 pagine per test rapidi\n• Usa 20+ pagine per analisi complete\n• 5-7 cluster sono ideali per la maggior parte dei casi")

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
                    recensioni_data = estrai_recensioni_con_metadata(url_base, max_pagine, progress_bar_1, status_text_1)
                
                if not recensioni_data:
                    st.error("❌ Nessuna recensione estratta. Verifica l'URL.")
                    return
                
                # Mostra statistiche rating
                st.markdown("### 📊 Distribuzione Rating")
                rating_counts = Counter(r['rating'] for r in recensioni_data)
                rating_cols = st.columns(5)
                for i, col in enumerate(rating_cols, 1):
                    count = rating_counts.get(i, 0)
                    col.metric(f"{i}⭐", count)
                
                # FASE 2: Clustering
                st.markdown("### 🎯 Fase 2: Clustering Tematico")
                with st.spinner("Clustering in corso..."):
                    recensioni_data, cluster_keywords = clusterizza_recensioni(recensioni_data, n_clusters)
                st.success(f"✅ Identificati {n_clusters} temi principali!")
                
                # Mostra cluster
                with st.expander("📊 Visualizza Temi Identificati"):
                    for cluster_id, keywords in cluster_keywords.items():
                        count = sum(1 for r in recensioni_data if r['cluster'] == cluster_id)
                        st.markdown(f"""
                        <div class="cluster-box">
                            <h4>Tema {cluster_id + 1} ({count} recensioni)</h4>
                            <p><strong>Parole chiave:</strong> {', '.join(keywords)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # FASE 3: Preparazione per AI
                st.markdown("### 📝 Fase 3: Preparazione per AI")
                testi_puliti = [r['testo_pulito'] for r in recensioni_data]
                testo_completo = " ".join(testi_puliti)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"📊 Creati {len(blocchi)} blocchi per l'analisi AI")
                
                # FASE 4: Analisi AI
                st.markdown("### 🤖 Fase 4: Analisi AI")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                with st.spinner("Analisi AI in corso..."):
                    risultati = analizza_blocchi_avanzata(blocchi, client, progress_bar_2, status_text_2)
                
                # FASE 5: Analisi Frequenze
                st.markdown("### 📈 Fase 5: Analisi Frequenze e Esempi")
                with st.spinner("Calcolo frequenze con AI..."):
                    analisi_frequenze = analizza_frequenze_temi_migliorata(risultati, recensioni_data, client)
                
                st.markdown('<div class="success-box"><h3>🎉 Analisi Completata!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI
                st.markdown("## 📊 Risultati Analisi")
                
                # Metriche
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("📝 Recensioni", len(recensioni_data))
                
                with col_m2:
                    st.metric("💪 Punti Forza", len(risultati['punti_forza']))
                
                with col_m3:
                    st.metric("⚠️ Criticità", len(risultati['punti_debolezza']))
                
                with col_m4:
                    st.metric("🎯 Temi", n_clusters)
                
                # Tabs per risultati
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "💪 Punti Forza", 
                    "⚠️ Criticità", 
                    "🎯 Leve Marketing", 
                    "📈 Strategie Digital", 
                    "🔍 Parole Chiave",
                    "📊 Cluster Tematici"
                ])
                
                with tab1:
                    st.markdown("### 💪 Punti di Forza con Frequenze")
                    
                    for punto, dati in list(analisi_frequenze['punti_forza'].items())[:10]:
                        col_forza1, col_forza2 = st.columns([3, 1])
                        
                        with col_forza1:
                            st.markdown(f"**{punto}**")
                        
                        with col_forza2:
                            st.markdown(f'<span class="frequency-badge">{dati["percentuale"]:.1f}% ({dati["count"]} menzioni)</span>', unsafe_allow_html=True)
                        
                        # Mostra esempi
                        if dati['esempi']:
                            with st.expander("📋 Vedi esempi"):
                                for esempio in dati['esempi']:
                                    rating_stars = "⭐" * esempio.get('rating', 0) if esempio.get('rating', 0) > 0 else ""
                                    st.markdown(f'{rating_stars}')
                                    st.markdown(f'<div class="review-example">"{esempio["testo"]}"</div>', unsafe_allow_html=True)
                                    if esempio['url'] != url_base:
                                        st.markdown(f"[🔗 Vai alla recensione]({esempio['url']})")
                
                with tab2:
                    st.markdown("### ⚠️ Punti di Debolezza con Frequenze")
                    
                    for punto, dati in list(analisi_frequenze['punti_debolezza'].items())[:10]:
                        col_deb1, col_deb2 = st.columns([3, 1])
                        
                        with col_deb1:
                            st.markdown(f"**{punto}**")
                        
                        with col_deb2:
                            st.markdown(f'<span class="frequency-badge" style="background: #dc3545;">{dati["percentuale"]:.1f}% ({dati["count"]} menzioni)</span>', unsafe_allow_html=True)
                        
                        # Mostra esempi
                        if dati['esempi']:
                            with st.expander("📋 Vedi esempi"):
                                for esempio in dati['esempi']:
                                    rating_stars = "⭐" * esempio.get('rating', 0) if esempio.get('rating', 0) > 0 else ""
                                    st.markdown(f'{rating_stars}')
                                    st.markdown(f'<div class="review-example">"{esempio["testo"]}"</div>', unsafe_allow_html=True)
                                    if esempio['url'] != url_base:
                                        st.markdown(f"[🔗 Vai alla recensione]({esempio['url']})")
                
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
                
                with tab6:
                    st.markdown("### 📊 Analisi dei Cluster Tematici")
                    
                    for cluster_id, keywords in cluster_keywords.items():
                        recensioni_cluster = [r for r in recensioni_data if r['cluster'] == cluster_id]
                        
                        st.markdown(f"""
                        <div class="cluster-box">
                            <h4>📌 Tema {cluster_id + 1}</h4>
                            <p><strong>Parole chiave:</strong> {', '.join(keywords)}</p>
                            <p><strong>Recensioni:</strong> {len(recensioni_cluster)} ({(len(recensioni_cluster)/len(recensioni_data)*100):.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostra esempio recensione per cluster
                        if recensioni_cluster:
                            with st.expander(f"Esempio recensione Tema {cluster_id + 1}"):
                                esempio = recensioni_cluster[0]
                                st.markdown(f'<div class="review-example">"{esempio["testo"][:300]}..."</div>', unsafe_allow_html=True)
                                if esempio['url'] != url_base:
                                    st.markdown(f"[🔗 Vai alla recensione]({esempio['url']})")
                
                # DOWNLOAD
                st.markdown("## 📥 Download Report")
                
                excel_data = crea_excel_download_avanzato(recensioni_data, risultati, analisi_frequenze, cluster_keywords)
                
                st.download_button(
                    label="📊 Scarica Report Excel Completo",
                    data=excel_data,
                    file_name="Analisi_Trustpilot_Report_Avanzato.xlsx",
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
        4. **Seleziona numero** di temi da identificare
        5. **Clicca** "Avvia Analisi"
        6. **Scarica** il report Excel
        
        ### 🆕 Nuove Funzionalità:
        • **Clustering tematico** delle recensioni
        • **Frequenze** per ogni punto forte/debole
        • **Esempi concreti** di recensioni
        • **Link diretti** alle recensioni originali
        • **Analisi quantitativa** dei temi
        
        ### 📈 Cosa Ottieni:
        • **Temi principali** delle recensioni
        • **Percentuali** di menzione
        • **Esempi reali** per ogni insight
        • **Strategie** basate sui dati
        """)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚡ Prestazioni</h4>
            <p>L'analisi richiede 3-7 minuti per 10 pagine con clustering</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Trustpilot Analyzer PRO v2.0 - Analisi avanzata con clustering e frequenze - Sviluppato da Daniele Pisciottano e il suo amico Claude 🦕</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
