import os
import streamlit as st
import anthropic
import requests

# 1. Set Page Configuration First
st.set_page_config(page_title="Multi-Agent Pre-Clinical Screening Engine", layout="wide")

# 2. Safe RDKit Import
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# 3. Initialize Anthropic Client Securely
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    st.error("⚠️ ANTHROPIC_API_KEY not found in Streamlit Secrets or Environment Variables.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# 4. Deterministic RDKit Property Engine
def compute_rdkit_properties(smiles_str):
    if not RDKIT_AVAILABLE:
        return None, "⚠️ RDKit is not installed in this Python environment."
    
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None, "⚠️ Invalid SMILES formula provided. Could not parse molecular structure."
        
        props = {
            "MW": round(Descriptors.MolWt(mol), 2),
            "cLogP": round(Crippen.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
            "HeavyAtoms": mol.GetNumHeavyAtoms()
        }
        
        formatted_text = f"""
        DETERMINISTIC RDKIT GROUND-TRUTH COMPUTATION:
        - Molecular Weight (MW): {props['MW']} Da
        - Calculated LogP (cLogP): {props['cLogP']}
        - Topological Polar Surface Area (TPSA): {props['TPSA']} Å²
        - Hydrogen Bond Donors (HBD): {props['HBD']}
        - Hydrogen Bond Acceptors (HBA): {props['HBA']}
        - Rotatable Bonds (RotB): {props['RotB']}
        - Fraction sp3 Carbons (Fsp3): {props['Fsp3']}
        - Heavy Atom Count: {props['HeavyAtoms']}
        """
        return props, formatted_text
    except Exception as e:
        return None, f"⚠️ Error computing RDKit properties: {e}"

# 5. Hardened Live External API Connectors (15s timeout with explicit UI errors)
def fetch_live_chembl_data(target_name):
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={target_name}&limit=1"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            targets = data.get('targets', [])
            if targets:
                target_chembl_id = targets[0].get('target_chembl_id')
                pref_name = targets[0].get('pref_name')
                return f"ChEMBL Target Match: {pref_name} ({target_chembl_id})"
        return f"No direct ChEMBL target match for '{target_name}' (Status code: {res.status_code})."
    except requests.exceptions.Timeout:
        return "⚠️ ChEMBL search timed out after 15 seconds."
    except Exception as e:
        return f"⚠️ ChEMBL search error: {e}"

def fetch_live_pubmed_data(target_name, indication_name):
    try:
        query = f"{target_name} {indication_name} mechanism OR target validation OR crystal structure"
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&pageSize=2"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            results = data.get('resultList', {}).get('result', [])
            literatures = []
            for item in results:
                title = item.get('title', 'No Title')
                journal = item.get('journalTitle', 'Journal N/A')
                pub_year = item.get('pubYear', 'N/A')
                literatures.append(f"- {title} ({journal}, {pub_year})")
            if literatures:
                return "LIVE PUBMED/EUROPE PMC MECHANISTIC LITERATURE:\n" + "\n".join(literatures)
        return "No recent target/indication mechanistic literature retrieved."
    except requests.exceptions.Timeout:
        return "⚠️ Europe PMC literature search timed out after 15 seconds."
    except Exception as e:
        return f"⚠️ Literature search error: {e}"

# 6. Establish Local Vector Store (RAG) with Provenance Support
DB_FOLDER = "./chroma_db"

@st.cache_resource
def init_vector_store():
    if os.path.exists(DB_FOLDER):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Vector store initialization warning: {e}")
            return None
    return None

VECTOR_DB = init_vector_store()

# 7. Refactored Neutral System Prompt with Base-Rate Calibration & Provenance Mandate
MULTI_AGENT_SYSTEM_PROMPT = """You are the AlphaForge Multi-Agent Pre-Clinical Evaluation Engine.
Act as an independent, rigorous, and objective **Drug Discovery & Medicinal Chemistry Review Panel**.

CRITICAL EVALUATION MANDATE:
Evaluate the candidate strictly from first principles using its chemical structure/SMILES, calculated RDKit properties, and target compartment biology. Your job is to rigorously test the asset and find reasons it could fail. Avoid sycophantic praise; evaluate risk with absolute neutrality.

PROVENANCE CITATION MANDATE:
When you integrate information or context from <local_vector_corpus_context>, you MUST explicitly cite the source filename (e.g., [Source: filename.pdf]) directly next to the claim or insight. Do not summarize from the corpus without attributing the exact source document.

Structure your response using the EXACT markdown headers below:

# 🔍 0. Prior-Knowledge Declaration
State explicitly whether you recognize this compound, scaffold, or target from published clinical or preclinical literature. If yes, state specifically what you recognize and from where. If no, state "No prior recognition".

# 📋 Executive Consensus & Phase-Gated Assessment

## 1. Consensus Verdict
State clearly: PROCEED, PROCEED WITH CAUTION, or REJECT, accompanied by an objective rationale.

## 2. Composite CDR Score
Provide an integer score from 0 to 100 (Lower is better).

## 3. Phase-Gated Rationale & Probabilities
Reference base rates for a small-molecule asset at this stage: Phase 1 ~65%, Phase 2 ~30%, Phase 3 ~58%. State each probability relative to its base rate and say explicitly whether your estimate is above, at, or below it, and why.
Provide a Markdown table:
| Phase | Assessment | Probability of Success | Rationale |
| Phase 1 (Safety/PK) | Favorable / Moderate / Unfavorable | XX% | Rationale citing RDKit properties & base-rate deviation |
| Phase 2 (PoC/Efficacy) | Favorable / Moderate / Unfavorable | XX% | Target validation, compartment accumulation & base-rate deviation |
| Phase 3 (Pivot/Redundancy) | Favorable / Moderate / Unfavorable | XX% | Biological durability, pathway compensation & base-rate deviation |

## 4. Executive Summary & Key Risk Matrix
Provide a Markdown table detailing key features and risks across MedChem, Toxicology, Pharmacology, and Target Biology.

---

# 🧬 Specialist Agent Panel Outputs

## 🧪 MedChem-Rationalist Agent Report
- **Structural Assessment:** Cite exact RDKit values (MW, cLogP, TPSA, HBD/HBA, RotB, Fsp3) and SMILES scaffold properties.
- **Compartment & Physicochemical Feasibility:** Passive permeability vs compartment trapping.
- **Metabolic Soft Spots & Proactive SAR Suggestions:** Provide 4–6 concrete bioisostere or structural optimization recommendations.

## ☣️ Toxi-Predictive-Toxicologist Agent Report
- **Off-Target Safety & Risk Mitigation:** Evaluate hERG, CAD, and phospholipidosis based on property profiles.
- **On-Target & Idiosyncratic Safety:** Predictable monitorable risks vs off-target margins.

## 💊 Pharma-Clinical-Pharmacologist Agent Report
- **Compartment Distribution & Exposure:** Target compartment accumulation and oral bioavailability.
- **Translational PK/PD Feasibility:** Biomarker readouts and DDI risks.

## 🔬 Biological-Rationalist Agent Report
- **Human Genetic Validation & Druggability:** OMIM/GWAS signals and MoA validation.
- **Biological Redundancy & Pathway Analysis:** Compensation mechanisms and trial design hurdles.
"""

def extract_text(response):
    text_chunks = []
    for block in response.content:
        if hasattr(block, 'text') and getattr(block, 'type', '') == 'text':
            text_chunks.append(block.text)
        elif hasattr(block, 'text') and not getattr(block, 'type', None):
            text_chunks.append(block.text)
    return "\n\n".join(text_chunks) if text_chunks else "No response text generated."

# 8. User Interface Panel
st.title("🔬 Multi-Agent Pre-Clinical Screening Engine")
st.subheader("Domain Specialist Assessment with Traceable Provenance & Prior-Knowledge Declaration")

if not RDKIT_AVAILABLE:
    st.warning("⚠️ RDKit Python package not detected. Install via `pip install rdkit` for exact property calculations.")

with st.form("drug_form"):
    smiles = st.text_input(
        "Chemical Structure (SMILES Formula):", 
        placeholder="Enter SMILES string..."
    )
    target = st.text_input(
        "Molecular Target:", 
        placeholder="Enter molecular target..."
    )
    indication = st.text_input(
        "Indication / Therapeutic Area:", 
        placeholder="Enter disease indication..."
    )
    submit = st.form_submit_button("Run Optimized Multi-Agent Matrix")

if submit:
    if not smiles or not target or not indication:
        st.error("⚠️ Please fill out all three fields before running the analysis.")
    else:
        rdkit_props, rdkit_text = compute_rdkit_properties(smiles)
        
        if rdkit_props is not None:
            st.success("✅ Deterministic RDKit Property Calculation Complete!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Exact MW", f"{rdkit_props['MW']} Da")
            col2.metric("Calculated LogP", f"{rdkit_props['cLogP']}")
            col3.metric("TPSA", f"{rdkit_props['TPSA']} Å²")
            col4.metric("Rotatable Bonds", f"{rdkit_props['RotB']}")

        with st.spinner("Connecting to Live ChEMBL & PubMed APIs..."):
            chembl_info = fetch_live_chembl_data(target)
            pubmed_info = fetch_live_pubmed_data(target, indication)

        context = ""
        if VECTOR_DB is not None:
            with st.spinner("Retrieving domain context from corpus with provenance..."):
                try:
                    query_string = (f"{target} {indication} kinase selectivity structural alert "
                                    f"reactive metabolite genetic validation pKa basicity "
                                    f"metabolic soft spot")
                    relevant_docs = VECTOR_DB.similarity_search_with_score(query_string, k=3)
                    
                    context_chunks = []
                    for doc, score in relevant_docs:
                        source_meta = doc.metadata.get("source", "Corpus Document")
                        context_chunks.append(f"[Source: {source_meta} | distance {score:.3f}]\n{doc.page_content}")
                    
                    context = "\n\n---\n\n".join(context_chunks)
                except Exception as e:
                    context = f"Vector search error: {e}"

        with st.spinner("Executing Multi-Agent Panel Analysis with Claude..."):
            try:
                combined_payload = f"""
<deterministic_properties>
{rdkit_text if rdkit_props else 'RDKit property calculation unavailable.'}
</deterministic_properties>

<external_database_context>
- {chembl_info}
{pubmed_info}
</external_database_context>

<local_vector_corpus_context>
{context}
</local_vector_corpus_context>

<user_target_evaluation>
[SMILES]: {smiles}
[Target]: {target}
[Indication]: {indication}
</user_target_evaluation>
"""

                response = client.messages.create(
                    model="claude-sonnet-5-5",
                    max_tokens=16000,
                    system=MULTI_AGENT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": combined_payload}]
                )
                
                if response.stop_reason == "max_tokens":
                    st.warning("⚠️ Warning: Output was truncated due to token limit constraints (max_tokens=16000).")

                output = extract_text(response)
                
                st.success("Analysis Complete!")
                st.markdown("---")
                st.markdown(output)

            except Exception as e:
                st.error(f"An error occurred during API execution: {e}")