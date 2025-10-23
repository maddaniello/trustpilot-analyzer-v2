# 🚀 ANALIZZATORE TRUSTPILOT - STREAMLIT APP
# Versione Web User-Friendly con Clustering e Sentiment Analysis

import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
        margin: 1rem 0;
    }
    
    .review-example {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    
    .positive-review {
        border-left: 4px solid #28a745;
    }
    
    .negative-review {
        border-left: 4px solid #dc3545;
    }
    
    .frequency-badge {
        background: #007bff;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 HEADER PRINCIPALE
st.markdown('<h1 class="main-header">🚀 Analizzatore Trustpilot Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Cosa fa questa App? (Versione Migliorata)</h3>
    <p>• Estrae automaticamente recensioni da Trustpilot con rating e link</p>
    <p>• Clusterizza le recensioni per tematiche comuni</p>
    <p>• Analizza la frequenza e il peso di ogni punto critico</p>
    <p>• Mostra esempi reali di recensioni positive/negative</p>
    <p>• Genera strategie di Digital Marketing personalizzate</p>
    <p>• Fornisce suggerimenti specifici per SEO, ADV, Email e CRO</p>
</div>
""", unsafe_allow_html=True)

# 🔧 FUNZIONI BACKEND MIGLIORATE
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
        "buono", "buona", "bene", "male", "servizio", "prodotto", "azienda", "sempre",
        # Aggiunti termini da escludere
        "verificata", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"
    ])

def pulisci_testo(testo):
    """Pulizia avanzata del testo"""
    stopwords_italiane = get_stopwords()
    testo = testo.lower()
    
    # Rimozione più aggressiva di date e termini temporali
    testo = re.sub(r'pubblicata il \d{1,2} \w+ \d{4}', '', testo)
    testo = re.sub(r'data dell\'esperienza:.*', '', testo, flags=re.IGNORECASE)
    testo = re.sub(r'\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    testo = re.sub(r'(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    
    # Rimozione parola "verificata" e varianti
    testo = re.sub(r'\bverificat[aoe]\b', '', testo)
    testo = re.sub(r'\bverifica\b', '', testo)
    
    # Rimozione stelle e rating
    testo = re.sub(r'stelle su 5', '', testo)
    testo = re.sub(r'\d+ stelle', '', testo)
    testo = re.sub(r'valutato \d', '', testo)
    
    # Pulizia generale
    testo = re.sub(r'[^\w\s]', ' ', testo)
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    
    parole = testo.split()
    parole_filtrate = [parola for parola in parole if parola not in stopwords_italiane and len(parola) > 2]
    return " ".join(parole_filtrate)

def estrai_rating_da_testo(testo):
    """Estrae il rating dalla recensione"""
    pattern_stelle = re.search(r'(\d)\s*stell[ae]', testo.lower())
    if pattern_stelle:
        return int(pattern_stelle.group(1))
    
    pattern_valutato = re.search(r'valutato\s*(\d)', testo.lower())
    if pattern_valutato:
        return int(pattern_valutato.group(1))
    
    return None

def estrai_recensioni_con_metadata(url_base, max_pagine, progress_bar, status_text):
    """Estrae recensioni con metadata (rating, link, data)"""
    tutte_recensioni = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Estrai il dominio dall'URL
    domain_match = re.search(r'/review/([^/?]+)', url_base)
    domain = domain_match.group(1) if domain_match else ""

    for pagina in range(1, max_pagine + 1):
        url_pagina = f"{url_base}?page={pagina}"
        status_text.text(f"🕷️ Estrazione pagina {pagina}/{max_pagine}...")
        
        try:
            response = requests.get(url_pagina, headers=headers, timeout=10)
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Cerca i container delle recensioni
            recensioni_containers = soup.find_all('article', {'data-service-review-rating': True})
            
            if not recensioni_containers:
                recensioni_containers = soup.select('section[class*="reviewContentwrapper"], div[class*="review-content"], article[class*="review"]')
            
            if not recensioni_containers and pagina == 1:
                st.error("❌ Nessuna recensione trovata. Verifica l'URL.")
                return []

            for container in recensioni_containers:
                try:
                    # Estrai testo
                    testo = container.get_text(separator=" ", strip=True)
                    if not testo or len(testo) < 50:
                        continue
                    
                    # Estrai rating
                    rating = None
                    rating_elem = container.get('data-service-review-rating')
                    if rating_elem:
                        rating = int(rating_elem)
                    else:
                        rating_div = container.find('div', {'data-service-review-rating': True})
                        if rating_div:
                            rating = int(rating_div.get('data-service-review-rating'))
                        else:
                            rating = estrai_rating_da_testo(testo)
                    
                    # Estrai ID recensione - proviamo diversi metodi
                    review_id = None
                    link = url_pagina  # Default fallback
                    
                    # Metodo 1: Cerca link con classe specifica che contiene l'ID recensione
                    review_link = container.find('a', {'data-review-url': True})
                    if review_link:
                        review_url = review_link.get('data-review-url')
                        if review_url:
                            review_id = review_url.split('/')[-1] if '/' in review_url else review_url
                    
                    # Metodo 2: Cerca nell'attributo data-service-review-id
                    if not review_id:
                        review_id = container.get('data-service-review-id')
                    
                    # Metodo 3: Cerca link "Date of experience" o simili che spesso contengono l'ID
                    if not review_id:
                        time_elem = container.find('time', {'datetime': True})
                        if time_elem:
                            parent_link = time_elem.find_parent('a', href=True)
                            if parent_link and '/reviews/' in parent_link['href']:
                                review_id = parent_link['href'].split('/reviews/')[-1].split('?')[0]
                    
                    # Metodo 4: Cerca qualsiasi link che contenga /reviews/
                    if not review_id:
                        all_links = container.find_all('a', href=True)
                        for link_elem in all_links:
                            href = link_elem['href']
                            if '/reviews/' in href:
                                # Estrai l'ID dalla URL
                                review_id = href.split('/reviews/')[-1].split('?')[0].split('#')[0]
                                break
                    
                    # Metodo 5: Cerca nel data-consumer-review-id
                    if not review_id:
                        review_id = container.get('data-consumer-review-id')
                    
                    # Costruisci il link corretto
                    if review_id:
                        # Formato corretto per link diretto alla singola recensione
                        link = f"https://it.trustpilot.com/reviews/{review_id}"
                    else:
                        # Fallback se non abbiamo l'ID
                        link = url_pagina
                    
                    # Estrai data se disponibile
                    data = None
                    data_match = re.search(r'(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})', testo)
                    if data_match:
                        data = data_match.group(1)
                    
                    recensione_data = {
                        'testo': testo,
                        'testo_pulito': pulisci_testo(testo),
                        'rating': rating,
                        'link': link,
                        'data': data,
                        'pagina': pagina,
                        'review_id': review_id  # Salviamo anche l'ID per uso futuro
                    }
                    
                    tutte_recensioni.append(recensione_data)
                    
                except Exception as e:
                    continue
            
            progress_bar.progress(pagina / max_pagine)
            time.sleep(1)

        except Exception as e:
            st.warning(f"⚠️ Errore pagina {pagina}: {e}")
            continue

    return tutte_recensioni

def clusterizza_recensioni(recensioni_data, n_clusters=None):
    """Clusterizza le recensioni per tematiche"""
    if len(recensioni_data) < 5:
        return recensioni_data, []
    
    # Prepara i testi per il clustering
    testi = [r['testo_pulito'] for r in recensioni_data]
    
    # Vectorizza i testi escludendo termini temporali
    vectorizer = TfidfVectorizer(
        max_features=100, 
        min_df=2, 
        max_df=0.8,
        # Aggiungi token_pattern per escludere numeri e date
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    try:
        X = vectorizer.fit_transform(testi)
    except:
        return recensioni_data, []
    
    # Determina numero ottimale di cluster se non specificato
    if n_clusters is None:
        n_clusters = min(8, max(3, len(recensioni_data) // 10))
    
    # Applica KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Aggiungi label ai dati
    for i, rec in enumerate(recensioni_data):
        rec['cluster'] = cluster_labels[i]
    
    # Estrai tematiche principali per ogni cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_topics = []
    recensioni_usate = set()  # Track delle recensioni già usate come esempio
    
    for i in range(n_clusters):
        # Trova le parole più rappresentative del cluster
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        # Filtra ulteriormente le parole per escludere termini non significativi
        top_words = [w for w in top_words if len(w) > 3 and not w.isdigit()]
        
        # Conta recensioni nel cluster
        cluster_reviews = [r for r in recensioni_data if r['cluster'] == i]
        
        # Seleziona recensioni esempio uniche per questo cluster
        esempi_cluster = []
        for rec in cluster_reviews:
            rec_id = rec.get('review_id', rec['testo'][:50])  # Usa ID o inizio testo come identificatore
            if rec_id not in recensioni_usate:
                esempi_cluster.append(rec)
                recensioni_usate.add(rec_id)
                if len(esempi_cluster) >= 3:  # Massimo 3 esempi per cluster
                    break
        
        cluster_info = {
            'id': i,
            'parole_chiave': top_words[:5],
            'n_recensioni': len(cluster_reviews),
            'percentuale': (len(cluster_reviews) / len(recensioni_data)) * 100,
            'rating_medio': np.mean([r['rating'] for r in cluster_reviews if r['rating']]),
            'recensioni': esempi_cluster  # Usa solo esempi unici
        }
        cluster_topics.append(cluster_info)
    
    # Ordina per numero di recensioni
    cluster_topics.sort(key=lambda x: x['n_recensioni'], reverse=True)
    
    return recensioni_data, cluster_topics

def analizza_frequenza_temi(risultati, recensioni_data):
    """Analizza la frequenza dei temi identificati nelle recensioni"""
    frequenze = {
        'punti_forza': {},
        'punti_debolezza': {}
    }
    
    recensioni_usate_forza = {}  # Track recensioni usate per punto di forza
    recensioni_usate_debolezza = {}  # Track recensioni usate per punto di debolezza
    
    # Calcola frequenze per punti di forza
    for punto in risultati['punti_forza']:
        count = 0
        esempi = []
        recensioni_ids_usate = set()
        
        # Estrai parole chiave significative dal punto (escludendo articoli e preposizioni)
        parole_punto = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec['rating'] and rec['rating'] >= 4:  # Solo recensioni positive
                rec_id = rec.get('review_id', rec['testo'][:50])
                
                # Verifica che la recensione non sia già stata usata per questo punto
                if rec_id not in recensioni_ids_usate:
                    # Cerca menzioni più precise del punto nel testo pulito
                    testo_lower = rec['testo_pulito'].lower()
                    
                    # Verifica presenza di almeno 2 parole chiave del punto
                    matches = sum(1 for parola in parole_punto if parola in testo_lower)
                    
                    if matches >= min(2, len(parole_punto)):
                        count += 1
                        recensioni_ids_usate.add(rec_id)
                        
                        # Aggiungi esempio solo se effettivamente contiene il tema
                        if len(esempi) < 2 and any(parola in testo_lower for parola in parole_punto):
                            esempi.append(rec)
        
        if count > 0:
            frequenze['punti_forza'][punto] = {
                'count': count,
                'percentuale': (count / len([r for r in recensioni_data if r['rating'] and r['rating'] >= 4])) * 100 if any(r['rating'] and r['rating'] >= 4 for r in recensioni_data) else 0,
                'esempi': esempi
            }
    
    # Calcola frequenze per punti di debolezza
    for punto in risultati['punti_debolezza']:
        count = 0
        esempi = []
        recensioni_ids_usate = set()
        
        # Estrai parole chiave significative dal punto
        parole_punto = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec['rating'] and rec['rating'] <= 2:  # Solo recensioni negative
                rec_id = rec.get('review_id', rec['testo'][:50])
                
                # Verifica che la recensione non sia già stata usata per questo punto
                if rec_id not in recensioni_ids_usate:
                    # Cerca menzioni più precise del punto nel testo pulito
                    testo_lower = rec['testo_pulito'].lower()
                    
                    # Verifica presenza di almeno 2 parole chiave del punto
                    matches = sum(1 for parola in parole_punto if parola in testo_lower)
                    
                    if matches >= min(2, len(parole_punto)):
                        count += 1
                        recensioni_ids_usate.add(rec_id)
                        
                        # Aggiungi esempio solo se effettivamente contiene il tema
                        if len(esempi) < 2 and any(parola in testo_lower for parola in parole_punto):
                            esempi.append(rec)
        
        if count > 0:
            frequenze['punti_debolezza'][punto] = {
                'count': count,
                'percentuale': (count / len([r for r in recensioni_data if r['rating'] and r['rating'] <= 2])) * 100 if any(r['rating'] and r['rating'] <= 2 for r in recensioni_data) else 0,
                'esempi': esempi
            }
    
    # Ordina per frequenza
    frequenze['punti_forza'] = dict(sorted(frequenze['punti_forza'].items(), 
                                          key=lambda x: x[1]['count'], reverse=True))
    frequenze['punti_debolezza'] = dict(sorted(frequenze['punti_debolezza'].items(), 
                                              key=lambda x: x[1]['count'], reverse=True))
    
    return frequenze

def analizza_blocchi_avanzata_con_sentiment(blocchi, client, progress_bar, status_text):
    """Analisi AI con sentiment analysis migliorata"""
    risultati = {
        "punti_forza": [],
        "punti_debolezza": [],
        "leve_marketing": [],
        "parole_chiave": [],
        "suggerimenti_seo": [],
        "suggerimenti_adv": [],
        "suggerimenti_email": [],
        "suggerimenti_cro": [],
        "suggerimenti_sinergie": [],
        "sentiment_distribution": {"positivo": 0, "neutro": 0, "negativo": 0}
    }

    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 Analizzando blocco {i+1}/{len(blocchi)} con AI...")

        prompt = f"""
        Analizza le seguenti recensioni di un'azienda e fornisci insights strategici dettagliati:

        RECENSIONI:
        {blocco}

        Rispondi SOLO in formato JSON valido con queste chiavi:
        {{
            "punti_forza": ["punto specifico 1", "punto specifico 2", ...],
            "punti_debolezza": ["problema specifico 1", "problema specifico 2", ...],
            "leve_marketing": ["leva concreta 1", "leva concreta 2", ...],
            "parole_chiave": ["termine rilevante 1", "termine rilevante 2", ...],
            "suggerimenti_seo": ["suggerimento SEO specifico 1", "suggerimento SEO specifico 2", ...],
            "suggerimenti_adv": ["strategia pubblicitaria 1", "strategia pubblicitaria 2", ...],
            "suggerimenti_email": ["strategia email 1", "strategia email 2", ...],
            "suggerimenti_cro": ["ottimizzazione conversioni 1", "ottimizzazione conversioni 2", ...],
            "suggerimenti_sinergie": ["sinergia tra canali 1", "sinergia tra canali 2", ...],
            "sentiment_counts": {{"positivo": N, "neutro": N, "negativo": N}}
        }}

        LINEE GUIDA IMPORTANTI:
        - Estrai punti MOLTO SPECIFICI, non generici
        - I punti di forza devono essere elementi concreti lodati dai clienti
        - I punti di debolezza devono essere problemi specifici lamentati
        - Identifica pattern ricorrenti nelle recensioni
        - Le leve marketing devono essere vantaggi competitivi unici
        - Conta approssimativamente il sentiment delle recensioni
        - SEO: suggerimenti per contenuti e keywords basati sui feedback reali
        - ADV: messaggi pubblicitari basati sui punti di forza reali
        - Email: automazioni basate sui comportamenti descritti
        - CRO: ottimizzazioni basate sui problemi UX menzionati
        - Sinergie: come combinare i canali per risolvere le criticità
        - IGNORA completamente date, mesi, giorni e la parola "verificata"
        - Non includere MAI termini temporali nelle parole chiave

        Fornisci suggerimenti PRATICI, SPECIFICI e IMPLEMENTABILI.
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
                        if chiave == "sentiment_counts":
                            # Aggrega i conteggi sentiment
                            for sent_type in ['positivo', 'neutro', 'negativo']:
                                if sent_type in dati['sentiment_counts']:
                                    risultati['sentiment_distribution'][sent_type] += dati['sentiment_counts'][sent_type]
                        else:
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

    # Deduplica mantenendo l'ordine di frequenza
    for chiave in risultati:
        if chiave != 'sentiment_distribution':
            risultati[chiave] = list(dict.fromkeys(risultati[chiave]))
    
    # Filtra ulteriormente parole chiave per rimuovere termini temporali
    termini_da_escludere = ['verificata', 'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 
                           'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
    risultati['parole_chiave'] = [
        parola for parola in risultati['parole_chiave'] 
        if parola.lower() not in termini_da_escludere and not any(t in parola.lower() for t in termini_da_escludere)
    ]
    
    return risultati

def mostra_esempi_recensioni(tema, esempi, tipo="positivo"):
    """Mostra esempi di recensioni per un tema specifico"""
    if not esempi:
        return
    
    st.markdown(f"**Esempi di recensioni:**")
    
    esempi_mostrati = set()  # Track degli esempi già mostrati
    
    for esempio in esempi[:2]:  # Mostra max 2 esempi
        # Crea un identificatore unico per evitare duplicati
        esempio_id = esempio.get('review_id', esempio['testo'][:50])
        
        if esempio_id not in esempi_mostrati:
            esempi_mostrati.add(esempio_id)
            
            rating_stars = "⭐" * (esempio['rating'] if esempio['rating'] else 3)
            
            # Estratto della recensione (primi 200 caratteri)
            testo_breve = esempio['testo'][:200] + "..." if len(esempio['testo']) > 200 else esempio['testo']
            
            css_class = "positive-review" if tipo == "positivo" else "negative-review"
            
            st.markdown(f"""
            <div class="review-example {css_class}">
                <div style="margin-bottom: 0.5rem;">
                    <strong>{rating_stars}</strong>
                    {f'<small style="color: #666;"> - {esempio["data"]}</small>' if esempio['data'] else ''}
                </div>
                <div style="margin-bottom: 0.5rem; color: #333;">
                    {testo_breve}
                </div>
                <a href="{esempio['link']}" target="_blank" style="color: #007bff; text-decoration: none;">
                    🔗 Vedi recensione completa su Trustpilot →
                </a>
            </div>
            """, unsafe_allow_html=True)

def crea_excel_download_avanzato(recensioni_data, risultati, clusters, frequenze):
    """Crea file Excel avanzato per download"""
    output = io.BytesIO()
    
    # DataFrame recensioni con metadata
    df_recensioni = pd.DataFrame([{
        'Testo Originale': r['testo'],
        'Rating': r['rating'],
        'Data': r['data'],
        'Cluster': r.get('cluster', ''),
        'Link': r['link']
    } for r in recensioni_data])
    
    # DataFrame clusters
    df_clusters = pd.DataFrame([{
        'Cluster ID': c['id'],
        'Tematica': ', '.join(c['parole_chiave']),
        'N. Recensioni': c['n_recensioni'],
        'Percentuale': f"{c['percentuale']:.1f}%",
        'Rating Medio': f"{c['rating_medio']:.1f}" if not np.isnan(c['rating_medio']) else 'N/A'
    } for c in clusters])
    
    # DataFrame frequenze punti di forza
    df_forza = pd.DataFrame([{
        'Punto di Forza': punto,
        'Frequenza': dati['count'],
        'Percentuale': f"{dati['percentuale']:.1f}%"
    } for punto, dati in frequenze['punti_forza'].items()])
    
    # DataFrame frequenze punti debolezza
    df_debolezza = pd.DataFrame([{
        'Punto di Debolezza': punto,
        'Frequenza': dati['count'],
        'Percentuale': f"{dati['percentuale']:.1f}%"
    } for punto, dati in frequenze['punti_debolezza'].items()])
    
    # DataFrame analisi generale
    df_analisi = pd.DataFrame({
        'Categoria': ['Leve Marketing', 'Parole Chiave'],
        'Insights': [
            ' | '.join(risultati['leve_marketing'][:10]),
            ' | '.join(risultati['parole_chiave'][:20])
        ]
    })
    
    # DataFrame strategie digital
    df_digital = pd.DataFrame({
        'Canale': ['SEO', 'ADV', 'Email Marketing', 'CRO', 'Sinergie'],
        'Suggerimenti Specifici': [
            ' | '.join(risultati['suggerimenti_seo']),
            ' | '.join(risultati['suggerimenti_adv']),
            ' | '.join(risultati['suggerimenti_email']),
            ' | '.join(risultati['suggerimenti_cro']),
            ' | '.join(risultati['suggerimenti_sinergie'])
        ]
    })
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_recensioni.to_excel(writer, sheet_name='Recensioni', index=False)
        df_clusters.to_excel(writer, sheet_name='Clusters Tematici', index=False)
        df_forza.to_excel(writer, sheet_name='Punti Forza Frequenza', index=False)
        df_debolezza.to_excel(writer, sheet_name='Punti Debolezza Frequenza', index=False)
        df_analisi.to_excel(writer, sheet_name='Analisi Generale', index=False)
        df_digital.to_excel(writer, sheet_name='Strategia Digital', index=False)
    
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
            max_value=100,
            value=10,
            help="Più pagine = più recensioni ma più tempo"
        )
        
        # Numero clusters
        n_clusters = st.slider(
            "🎯 Numero di cluster tematici",
            min_value=3,
            max_value=15,
            value=8,
            help="Numero di tematiche in cui raggruppare le recensioni"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Suggerimenti")
        st.info("• Inizia con 5-10 pagine per test rapidi\n• Usa 20+ pagine per analisi complete\n• 6-10 cluster sono ideali per la maggior parte dei casi\n• Ogni pagina ha ~20 recensioni")

    # AREA PRINCIPALE
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Inizia l'Analisi")
        
        if st.button("🔍 Avvia Analisi Avanzata Trustpilot", type="primary"):
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
                # FASE 1: Estrazione con metadata
                st.markdown("### 🕷️ Fase 1: Estrazione Recensioni con Metadata")
                progress_bar_1 = st.progress(0)
                status_text_1 = st.empty()
                
                with st.spinner("Estrazione recensioni e metadata in corso..."):
                    recensioni_data = estrai_recensioni_con_metadata(url_base, max_pagine, progress_bar_1, status_text_1)
                
                if not recensioni_data:
                    st.error("❌ Nessuna recensione estratta. Verifica l'URL.")
                    return
                
                # Statistiche recensioni
                n_con_rating = len([r for r in recensioni_data if r['rating']])
                rating_medio = np.mean([r['rating'] for r in recensioni_data if r['rating']])
                
                st.success(f"✅ Estratte {len(recensioni_data)} recensioni!")
                st.info(f"📊 {n_con_rating} recensioni con rating • Rating medio: {rating_medio:.1f} ⭐")
                
                # FASE 2: Clustering
                st.markdown("### 🎨 Fase 2: Clustering Tematico")
                with st.spinner("Clustering delle recensioni in corso..."):
                    recensioni_data, clusters = clusterizza_recensioni(recensioni_data, n_clusters)
                
                st.success(f"✅ Identificati {len(clusters)} cluster tematici!")
                
                # Mostra preview clusters
                with st.expander("👀 Vedi Cluster Tematici"):
                    for cluster in clusters[:5]:
                        st.markdown(f"""
                        <div class="cluster-box">
                            <strong>Cluster {cluster['id'] + 1}</strong>
                            <span class="frequency-badge">{cluster['percentuale']:.1f}% delle recensioni</span>
                            <br>
                            <small>Tematiche: {', '.join(cluster['parole_chiave'])}</small>
                            <br>
                            <small>Rating medio: {cluster['rating_medio']:.1f} ⭐</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # FASE 3: Preparazione per AI
                st.markdown("### 📝 Fase 3: Preparazione per Analisi AI")
                recensioni_pulite = [r['testo_pulito'] for r in recensioni_data]
                testo_completo = " ".join(recensioni_pulite)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"📊 Creati {len(blocchi)} blocchi per l'analisi AI")
                
                # FASE 4: Analisi AI con Sentiment
                st.markdown("### 🤖 Fase 4: Analisi AI con Sentiment Analysis")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                with st.spinner("Analisi AI avanzata in corso..."):
                    risultati = analizza_blocchi_avanzata_con_sentiment(blocchi, client, progress_bar_2, status_text_2)
                
                # FASE 5: Analisi Frequenze
                st.markdown("### 📊 Fase 5: Analisi Frequenze e Correlazioni")
                with st.spinner("Analisi delle frequenze..."):
                    frequenze = analizza_frequenza_temi(risultati, recensioni_data)
                
                st.markdown('<div class="success-box"><h3>🎉 Analisi Completata!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI AVANZATI
                st.markdown("## 📊 Risultati Analisi Avanzata")
                
                # Metriche principali
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                
                with col_m1:
                    st.metric("📝 Recensioni", len(recensioni_data))
                
                with col_m2:
                    st.metric("⭐ Rating Medio", f"{rating_medio:.1f}")
                
                with col_m3:
                    st.metric("💪 Punti Forza", len(risultati['punti_forza']))
                
                with col_m4:
                    st.metric("⚠️ Criticità", len(risultati['punti_debolezza']))
                
                with col_m5:
                    st.metric("🎯 Cluster", len(clusters))
                
                # Distribuzione sentiment
                if risultati['sentiment_distribution']['positivo'] > 0:
                    st.markdown("### 😊 Distribuzione Sentiment")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    total_sentiment = sum(risultati['sentiment_distribution'].values())
                    
                    with col_s1:
                        perc = (risultati['sentiment_distribution']['positivo'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😊 Positivo", f"{perc:.1f}%")
                    
                    with col_s2:
                        perc = (risultati['sentiment_distribution']['neutro'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😐 Neutro", f"{perc:.1f}%")
                    
                    with col_s3:
                        perc = (risultati['sentiment_distribution']['negativo'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😞 Negativo", f"{perc:.1f}%")
                
                # Tabs per risultati dettagliati
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "💪 Punti Forza", 
                    "⚠️ Criticità", 
                    "🎨 Cluster", 
                    "🎯 Leve Marketing", 
                    "📈 Strategie Digital", 
                    "🔍 Parole Chiave"
                ])
                
                with tab1:
                    st.markdown("### 💪 Punti di Forza (Ordinati per Frequenza)")
                    
                    if frequenze['punti_forza']:
                        for punto, dati in list(frequenze['punti_forza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge">Presente nel {dati['percentuale']:.1f}% delle recensioni positive</span>
                            """, unsafe_allow_html=True)
                            
                            if dati['esempi']:
                                with st.expander(f"Vedi esempi ({len(dati['esempi'])} disponibili)"):
                                    mostra_esempi_recensioni(punto, dati['esempi'], tipo="positivo")
                    else:
                        # Fallback se non ci sono frequenze calcolate
                        for i, punto in enumerate(risultati['punti_forza'][:10], 1):
                            st.markdown(f"**{i}.** {punto}")
                
                with tab2:
                    st.markdown("### ⚠️ Punti di Debolezza (Ordinati per Frequenza)")
                    
                    if frequenze['punti_debolezza']:
                        for punto, dati in list(frequenze['punti_debolezza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge" style="background: #dc3545;">Presente nel {dati['percentuale']:.1f}% delle recensioni negative</span>
                            """, unsafe_allow_html=True)
                            
                            if dati['esempi']:
                                with st.expander(f"Vedi esempi ({len(dati['esempi'])} disponibili)"):
                                    mostra_esempi_recensioni(punto, dati['esempi'], tipo="negativo")
                    else:
                        # Fallback se non ci sono frequenze calcolate
                        for i, punto in enumerate(risultati['punti_debolezza'][:10], 1):
                            st.markdown(f"**{i}.** {punto}")
                
                with tab3:
                    st.markdown("### 🎨 Analisi Cluster Tematici")
                    
                    esempi_mostrati_globali = set()  # Track globale degli esempi mostrati
                    
                    for cluster in clusters:
                        with st.expander(f"Cluster {cluster['id'] + 1}: {', '.join(cluster['parole_chiave'][:3])} ({cluster['percentuale']:.1f}%)"):
                            col_c1, col_c2 = st.columns(2)
                            
                            with col_c1:
                                st.metric("Recensioni", cluster['n_recensioni'])
                                st.metric("Rating Medio", f"{cluster['rating_medio']:.1f} ⭐")
                            
                            with col_c2:
                                st.markdown("**Tematiche principali:**")
                                for parola in cluster['parole_chiave']:
                                    st.markdown(f"• {parola}")
                            
                            # Mostra alcune recensioni esempio del cluster (uniche)
                            st.markdown("**Recensioni esempio di questo cluster:**")
                            esempi_mostrati = 0
                            for rec in cluster['recensioni']:
                                rec_id = rec.get('review_id', rec['testo'][:50])
                                if rec_id not in esempi_mostrati_globali and esempi_mostrati < 2:
                                    esempi_mostrati_globali.add(rec_id)
                                    esempi_mostrati += 1
                                    rating_stars = "⭐" * (rec['rating'] if rec['rating'] else 3)
                                    testo_breve = rec['testo'][:150] + "..." if len(rec['testo']) > 150 else rec['testo']
                                    st.markdown(f"> {rating_stars} {testo_breve}")
                                    st.markdown(f"[🔗 Vedi recensione completa]({rec['link']})")
                
                with tab4:
                    st.markdown("### 🎯 Leve Marketing Strategiche")
                    for i, leva in enumerate(risultati['leve_marketing'][:10], 1):
                        st.markdown(f"**{i}.** {leva}")
                
                with tab5:
                    st.markdown("### 📈 Strategie Digital Marketing")
                    
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
                
                with tab6:
                    st.markdown("### 🔍 Parole Chiave Principali")
                    parole_cols = st.columns(3)
                    for i, parola in enumerate(risultati['parole_chiave'][:15]):
                        with parole_cols[i % 3]:
                            st.markdown(f"🔸 **{parola}**")
                
                # DOWNLOAD AVANZATO
                st.markdown("## 📥 Download Report Avanzato")
                
                excel_data = crea_excel_download_avanzato(recensioni_data, risultati, clusters, frequenze)
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.download_button(
                        label="📊 Scarica Report Excel Completo",
                        data=excel_data,
                        file_name="Analisi_Trustpilot_Avanzata.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                
                with col_d2:
                    # Crea report JSON per elaborazioni future
                    json_report = {
                        'metadata': {
                            'n_recensioni': len(recensioni_data),
                            'rating_medio': float(rating_medio) if not np.isnan(rating_medio) else None,
                            'n_clusters': len(clusters),
                            'url_analizzato': url_base
                        },
                        'clusters': [{
                            'id': c['id'],
                            'tematiche': c['parole_chiave'],
                            'percentuale': c['percentuale'],
                            'rating_medio': float(c['rating_medio']) if not np.isnan(c['rating_medio']) else None
                        } for c in clusters],
                        'insights': risultati,
                        'frequenze': {
                            'punti_forza': {k: {'count': v['count'], 'percentuale': v['percentuale']} 
                                          for k, v in list(frequenze['punti_forza'].items())[:20]},
                            'punti_debolezza': {k: {'count': v['count'], 'percentuale': v['percentuale']} 
                                              for k, v in list(frequenze['punti_debolezza'].items())[:20]}
                        }
                    }
                    
                    st.download_button(
                        label="💾 Scarica Report JSON",
                        data=json.dumps(json_report, indent=2, ensure_ascii=False),
                        file_name="Analisi_Trustpilot_Data.json",
                        mime="application/json"
                    )
    
    with col2:
        st.markdown("## 📋 Guida Rapida")
        
        st.markdown("""
        ### 🆕 Nuove Funzionalità:
        - **🎨 Clustering Tematico**: Raggruppa recensioni per argomento
        - **📊 Analisi Frequenze**: Mostra quanto spesso appaiono i temi
        - **👁️ Esempi Reali**: Vedi recensioni esempio per ogni punto
        - **🔗 Link Diretti**: Accedi alle recensioni su Trustpilot
        - **😊 Sentiment Analysis**: Distribuzione emotiva delle recensioni
        
        ### 🎯 Come Usare:
        1. **Inserisci API Key** OpenAI nella sidebar
        2. **Copia URL** della pagina Trustpilot
        3. **Scegli numero** di pagine e cluster
        4. **Clicca** "Avvia Analisi Avanzata"
        5. **Esplora** i risultati interattivi
        6. **Scarica** il report completo
        
        ### 📈 Output Migliorati:
        • **Cluster tematici** con percentuali
        • **Frequenza** di ogni criticità/forza
        • **Esempi concreti** di recensioni
        • **Link diretti** a Trustpilot
        • **Rating medio** per cluster
        • **Sentiment distribution**
        • **Report JSON** per integrazioni
        """)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚡ Performance</h4>
            <p>L'analisi avanzata richiede 3-6 minuti per 10 pagine</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Trustpilot Analyzer PRO v2.1 - Analisi Avanzata con Clustering e Sentiment Analysis</p>
        <p>Sviluppato da Daniele Pisciottano e Claude 🦕</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
