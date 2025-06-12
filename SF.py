from transformers import pipeline
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Modelli Transformers
generator = pipeline("text2text-generation", model="t5-small")  # 1. Genera la query
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
ner = pipeline(
    "ner",
    model="xlm-roberta-large-finetuned-conll03-english",
    aggregation_strategy="simple"
)

# Genera una query Google a partire da input
def genera_query(input_text):
    result = generator(f"Generate a detailed web search query about: {input_text}", max_length=128, do_sample=True, temperature=0.9)
    return result[0]['generated_text']

# ricerca su Google
def cerca_siti(query, num=5):
    return list(search(query, num_results=num, lang="it"))

# Modello per similarity
sim_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

def è_coerente(query, testo, soglia=0.60):
    emb_query = sim_model.encode(query, convert_to_tensor=True)
    emb_testo = sim_model.encode(testo[:1024], convert_to_tensor=True)
    similarità = util.pytorch_cos_sim(emb_query, emb_testo).item()
    print(f"📏 Similarità contenuto: {similarità:.2f}")
    return similarità >= soglia

# Estrai testo HTML
def estrai_testo(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Per evitare blocchi da bot
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "header", "footer", "nav", "form", "noscript", "aside"]):
            tag.decompose()

        testo = soup.get_text(separator=' ', strip=True)
        return testo if testo.strip() else "[Nessun contenuto leggibile]"
    except Exception as e:
        return f"[Errore durante il parsing: {e}]"

# Esegui NER
def estrai_ner(testo):
    return ner(testo[:1024])

def estrai_ner_esteso(testo, blocco_caratteri=512):
    blocchi = [testo[i:i+blocco_caratteri] for i in range(0, len(testo), blocco_caratteri)]
    entita_totali = []
    for blocco in blocchi[:10]:  # Limita a 10 blocchi per sicurezza (puoi cambiare)
        entita_blocco = ner(blocco)
        entita_totali.extend(entita_blocco)
    return entita_totali

def crea_grafo_entita(entita_estratte, testo):
    # Inizializza grafo
    G = nx.Graph()
    
    # Filtra solo entità utili
    entita_rilevanti = [e for e in entita_estratte if e['entity_group'] in ['PER', 'ORG', 'LOC']]
    
    # Crea nodi per ogni entità
    for ent in entita_rilevanti:
        G.add_node(ent['word'], tipo=ent['entity_group'])
    
    # Trova co-occorrenze in frasi
    frasi = re.split(r'[.!?]', testo[:2048])
    for frase in frasi:
        presenti = {e['word'] for e in entita_rilevanti if e['word'] in frase}
        for e1 in presenti:
            for e2 in presenti:
                if e1 != e2:
                    if G.has_edge(e1, e2):
                        G[e1][e2]['weight'] += 1
                    else:
                        G.add_edge(e1, e2, weight=1)
    
    # Disegna grafo
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    colori = {'PER': 'skyblue', 'ORG': 'orange', 'LOC': 'lightgreen'}
    node_colors = [colori.get(G.nodes[n]['tipo'], 'gray') for n in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', font_size=10, node_size=1500)
    plt.title("📊 Grafo delle Entità NER")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === MAIN FLOW ===
def analizza(input_text):
    print(f"\n📌 Input: {input_text}")
    query = genera_query(input_text)
    print(f"🔍 Query generata: {query}\n")

    urls = cerca_siti(query)
    for url in urls:
        print(f"🌐 Sito trovato: {url}")
        testo = estrai_testo(url)
        print("🧪 Valutazione rilevanza:")
        if è_coerente(input_text, testo):
            print("✅ Il contenuto è coerente con la ricerca.")
            print("\n📖 Contenuto estratto:")
            if testo and isinstance(testo, str) and len(testo.strip()) > 50:
                if è_coerente(input_text, testo):
                    ...
            else:
                print("⚠️ Nessun contenuto utile da analizzare in questo sito.\n")
            print(testo[:500], "...\n")

            print("🔎 Entità trovate (NER):")
            entita = estrai_ner_esteso(testo)
            for ent in entita:
                print(f"- {ent['word']} ({ent['entity_group']})")

            # ➕ Visualizza grafo
            #crea_grafo_entita(entita, testo)

            break  # Si ferma al primo sito buono

        else:
            print("⛔ Sito non rilevante.\n")

# === ESECUZIONE ===
if __name__ == "__main__":
    print("💬 Inserisci un nome o argomento da cercare (scrivi 'exit' per uscire)")
    while True:
        query_input = input("\n🔎 Cerca: ")
        if query_input.lower() in ["exit", "esci", "quit"]:
            print("👋 Fine.")
            break
        analizza(query_input)