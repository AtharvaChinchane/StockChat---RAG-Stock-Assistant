

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from fuzzywuzzy import process
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import logging
from scipy.stats import pearsonr

# Neo4j imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    st.warning("Neo4j driver not installed. Install with: pip install neo4j")

# Import your actual database functions
from utils.mysql_helper import (
    create_user, authenticate_user, get_watchlist,
    add_to_watchlist, remove_from_watchlist
)

# --- Page Config ---
st.set_page_config(
    page_title="StockChat",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- Neo4j Integration Classes ---
class StockGraphDB:
    def __init__(self, uri, user, password):
        if not NEO4J_AVAILABLE:
            raise Exception("Neo4j driver not available")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def test_connection(self):
        """Test the Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logging.error(f"Neo4j connection test failed: {e}")
            return False
    
    def create_stock_node(self, ticker, name, sector, market_cap, currency):
        """Create or update a stock node"""
        with self.driver.session() as session:
            session.run("""
                MERGE (s:Stock {ticker: $ticker})
                SET s.name = $name,
                    s.sector = $sector,
                    s.market_cap = $market_cap,
                    s.currency = $currency,
                    s.updated_at = datetime()
                """, 
                ticker=ticker, name=name, sector=sector, 
                market_cap=market_cap, currency=currency
            )
    
    def create_correlation_relationship(self, ticker1, ticker2, correlation, period="3mo"):
        """Create correlation relationship between two stocks"""
        with self.driver.session() as session:
            session.run("""
                MATCH (s1:Stock {ticker: $ticker1})
                MATCH (s2:Stock {ticker: $ticker2})
                MERGE (s1)-[r:CORRELATES_WITH {period: $period}]-(s2)
                SET r.correlation = $correlation,
                    r.updated_at = datetime()
                """,
                ticker1=ticker1, ticker2=ticker2, 
                correlation=correlation, period=period
            )
    
    def create_sector_relationship(self, ticker, sector):
        with self.driver.session() as session:
            session.run("""
                MERGE (s:Sector {name: $sector})
                WITH s
                MATCH (stock:Stock {ticker: $ticker})
                MERGE (stock)-[:BELONGS_TO]->(s)
                """,
                ticker=ticker, sector=sector
            )
    def get_correlated_stocks(self, ticker, min_correlation=0.7, limit=10):
        """Get stocks correlated with the given ticker"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Stock {ticker: $ticker})-[r:CORRELATES_WITH]-(s2:Stock)
                WHERE r.correlation >= $min_correlation
                RETURN s2.ticker as ticker, s2.name as name, 
                       r.correlation as correlation
                ORDER BY r.correlation DESC
                LIMIT $limit
                """,
                ticker=ticker, min_correlation=min_correlation, limit=limit
            )
            return [dict(record) for record in result]
    
    def get_sector_stocks(self, sector, limit=20):
        """Get all stocks in a sector"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector {name: $sector})
                RETURN s.ticker as ticker, s.name as name, s.market_cap as market_cap
                ORDER BY s.market_cap DESC
                LIMIT $limit
                """,
                sector=sector, limit=limit
            )
            return [dict(record) for record in result]
    
    def get_portfolio_risk_analysis(self, tickers):
        """Analyze portfolio concentration risk"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
                WHERE s.ticker IN $tickers
                RETURN sec.name as sector, count(s) as stock_count,
                       collect(s.ticker) as stocks
                ORDER BY stock_count DESC
                """,
                tickers=tickers
            )
            return [dict(record) for record in result]
    
    def find_diversification_opportunities(self, user_tickers, exclude_sectors=None):
        """Suggest stocks for portfolio diversification"""
        exclude_sectors = exclude_sectors or []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
                WHERE NOT sec.name IN $exclude_sectors
                AND NOT s.ticker IN $user_tickers
                WITH sec, s
                ORDER BY s.market_cap DESC
                RETURN sec.name as sector, collect(s.ticker)[0..3] as top_stocks
                LIMIT 5
                """,
                exclude_sectors=exclude_sectors, user_tickers=user_tickers
            )
            return [dict(record) for record in result]

class StockCorrelationAnalyzer:
    def __init__(self, neo4j_db):
        self.db = neo4j_db
    
    def calculate_correlation_matrix(self, tickers, period="3mo"):
        """Calculate correlation matrix for given tickers"""
        try:
            # Fetch historical data
            data = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    if not hist.empty and len(hist) > 10:  # Need sufficient data
                        data[ticker] = hist['Close'].pct_change().dropna()
                except Exception as e:
                    logging.error(f"Error fetching data for {ticker}: {e}")
                    continue
            
            # Calculate correlations
            correlations = []
            tickers_list = list(data.keys())
            
            for i, ticker1 in enumerate(tickers_list):
                for j, ticker2 in enumerate(tickers_list):
                    if i < j:  # Avoid duplicates
                        try:
                            # Align data by date
                            common_dates = data[ticker1].index.intersection(data[ticker2].index)
                            if len(common_dates) > 10:  # Need sufficient overlapping data
                                series1 = data[ticker1].loc[common_dates]
                                series2 = data[ticker2].loc[common_dates]
                                
                                corr, p_value = pearsonr(series1, series2)
                                if not np.isnan(corr) and abs(corr) > 0.1:  # Only meaningful correlations
                                    correlations.append({
                                        'ticker1': ticker1,
                                        'ticker2': ticker2,
                                        'correlation': corr
                                    })
                        except Exception as e:
                            logging.error(f"Error calculating correlation between {ticker1} and {ticker2}: {e}")
                            continue
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating correlations: {e}")
            return []
    
    def update_stock_relationships(self, tickers):
        """Update stock data and relationships in Neo4j"""
        updated_count = 0
        
        # First, update stock nodes
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                self.db.create_stock_node(
                    ticker=ticker,
                    name=info.get('longName', info.get('shortName', ticker)),
                    sector=info.get('sector', 'Unknown'),
                    market_cap=info.get('marketCap', 0),
                    currency='INR' if '.NS' in ticker else 'USD'
                )
                
                # Create sector relationship
                if info.get('sector'):
                    self.db.create_sector_relationship(ticker, info.get('sector'))
                
                updated_count += 1
                    
            except Exception as e:
                logging.error(f"Error updating stock {ticker}: {e}")
        
        # Calculate and store correlations
        if len(tickers) > 1:
            correlations = self.calculate_correlation_matrix(tickers)
            for corr in correlations:
                try:
                    self.db.create_correlation_relationship(
                        corr['ticker1'], corr['ticker2'], corr['correlation']
                    )
                except Exception as e:
                    logging.error(f"Error storing correlation: {e}")
        
        return updated_count

# --- RAG System Setup ---
@st.cache_resource
def load_sentence_transformer():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_faq_data(file_path):
    """Load FAQ data from text file with multi-line answers"""
    if not os.path.exists(file_path):
        st.error(f"FAQ file does not exist: {file_path}")
        return []
    
    faq_questions = []
    faq_answers = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_q = None
        current_a = []

        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                if current_q and current_a:
                    # Save previous Q&A
                    faq_questions.append(current_q)
                    faq_answers.append(" ".join(current_a))
                current_q = line[2:].strip()
                current_a = []
            elif line.startswith("A:"):
                current_a.append(line[2:].strip())
            else:
                if line:  # continuation of answer
                    current_a.append(line)

        # Add the last Q&A
        if current_q and current_a:
            faq_questions.append(current_q)
            faq_answers.append(" ".join(current_a))

        # Convert to list of dicts with 'combined' field for embeddings
        qa_pairs = []
        for q, a in zip(faq_questions, faq_answers):
            qa_pairs.append({
                'question': q,
                'answer': a,
                'combined': f"Q: {q} A: {a}"
            })

        return qa_pairs

    except Exception as e:
        st.error(f"Error loading FAQ file: {e}")
        return []

def create_embeddings(qa_pairs, model):
    """Create embeddings for all Q&A pairs"""
    texts = [qa['combined'] for qa in qa_pairs]
    embeddings = model.encode(texts)
    return embeddings

def find_best_answer(query, qa_pairs, embeddings, model, threshold=0.3):
    """Find the best matching answer for a query"""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    if best_similarity >= threshold:
        return qa_pairs[best_match_idx], best_similarity
    else:
        return None, best_similarity

# Initialize RAG components
@st.cache_data
def initialize_rag_system():
    """Initialize the RAG system"""
    faq_file_path = r"A:\projects\stock chatbot\faq_data.txt"  # Update this path
    qa_pairs = load_faq_data(faq_file_path)
    
    if not qa_pairs:
        return None, None, None
    
    model = load_sentence_transformer()
    embeddings = create_embeddings(qa_pairs, model)
    
    return qa_pairs, embeddings, model

# --- Neo4j Connection Setup ---
@st.cache_resource
def init_neo4j_connection():
    """Initialize Neo4j connection"""
    if not NEO4J_AVAILABLE:
        return None, None, False
    
    # Neo4j configuration - Update these values
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your password
    
    try:
        stock_db = StockGraphDB(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if stock_db.test_connection():
            analyzer = StockCorrelationAnalyzer(stock_db)
            return stock_db, analyzer, True
        else:
            return None, None, False
    except Exception as e:
        logging.error(f"Neo4j connection failed: {str(e)}")
        return None, None, False

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .faq-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .faq-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .faq-answer {
        background: #4e4ed7;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .neo4j-status {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .neo4j-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .neo4j-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
    
    .index-card {
        background: linear-gradient(135deg, #4e4ed7 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .index-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .index-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .index-price {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    
    .index-change {
        font-size: 1rem;
        font-weight: 600;
    }
    
    .positive-change {
        color: #27ae60 !important;
    }
    
    .negative-change {
        color: #e74c3c !important;
    }
    
    .watchlist-container {
        background: #4e4ed7;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .watchlist-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .watchlist-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .watchlist-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .watchlist-ticker {
        font-size: 0.9rem;
        color: #7f8c8d;
        background: #ecf0f1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    
    .watchlist-price {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0.3rem 0;
    }
    
    .watchlist-change {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    
    .price-up {
        background: #d5f4e6;
        color: #27ae60;
    }
    
    .price-down {
        background: #fdf2f2;
        color: #e74c3c;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #4e4ed7 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
    }
    
    .success-msg {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .error-msg {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: #4e4ed7;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .correlation-container {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# --- Popular Stock Tickers ---
POPULAR_STOCKS = {
    "Reliance":"RELIANCE.NS", "Reliance Industries":"RELIANCE.NS",
    "Tata Consultancy":"TCS.NS", "Tata Consultancy Services":"TCS.NS",
    "HDFC Bank":"HDFCBANK.NS", 
    "Infosys":"INFY.NS", "Infy":"INFY.NS",
    "ICICI Bank":"ICICIBANK.NS",
    "Hindustan Unilever":"HINDUNILVR.NS",
    "Kotak Mahindra Bank":"KOTAKBANK.NS", "Kotak":"KOTAKBANK.NS",
    "Larsen & Toubro":"LT.NS", 
    "State Bank of India":"SBIN.NS", "SBI":"SBIN.NS",
    "Maruti Suzuki":"MARUTI.NS", "Maruti":"MARUTI.NS",
    "Axis Bank":"AXISBANK.NS",
    "HCL Tech":"HCLTECH.NS", "HCL Technologies":"HCLTECH.NS",
    "Bharti Airtel":"BHARTIARTL.NS", "Airtel":"BHARTIARTL.NS",
    "Adani Enterprises":"ADANIENT.NS", "Adani":"ADANIENT.NS",
    "Bajaj Finance":"BAJFINANCE.NS", "Bajaj Finserv":"BAJAJFINSV.NS",
    "Asian Paints":"ASIANPAINT.NS",
    "Tata Steel":"TATASTEEL.NS",
    "Apple":"AAPL",
    "Microsoft":"MSFT",
    "Amazon":"AMZN",
    "Tesla":"TSLA",
    "Alphabet":"GOOGL", "Google":"GOOGL",
    "Meta":"META", "Facebook":"META",
    "NVIDIA":"NVDA"
}

# --- Session State ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "auth_success" not in st.session_state:
    st.session_state.auth_success = False
if "neo4j_connected" not in st.session_state:
    st.session_state.neo4j_connected = False
if "stock_db" not in st.session_state:
    st.session_state.stock_db = None
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None

# Initialize RAG System
try:
    qa_pairs, embeddings, model = initialize_rag_system()
    rag_available = qa_pairs is not None
except Exception as e:
    st.warning(f"RAG system initialization failed: {str(e)}")
    rag_available = False
    qa_pairs, embeddings, model = None, None, None

# Initialize Neo4j System
if not st.session_state.neo4j_connected and NEO4J_AVAILABLE:
    stock_db, analyzer, connected = init_neo4j_connection()
    if connected:
        st.session_state.stock_db = stock_db
        st.session_state.analyzer = analyzer
        st.session_state.neo4j_connected = True

# --- Helper Functions ---
def get_ticker_from_input(user_input):
    """Convert user input to a ticker symbol using fuzzy matching."""
    user_input = user_input.strip()

    # Priority 1: Check if it's already a valid ticker format
    if '.' in user_input or user_input.isupper() and len(user_input) < 6:
        return user_input.upper()

    # Priority 2: Find the best match from popular stocks using fuzzywuzzy
    best_match, score = process.extractOne(user_input.lower(), POPULAR_STOCKS.keys())

    if score >= 85:
        return POPULAR_STOCKS[best_match]
    
    # Priority 3: If no good match is found, fallback to the original assumption
    st.warning(f"Could not find a strong match for '{user_input}'. Trying a direct lookup.")
    return user_input.upper() + ".NS"

def get_stock_info(ticker):
    """Get comprehensive stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        return {
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', ticker)),
            'price': current_price,
            'change': change,
            'change_pct': change_pct,
            'currency': 'INR' if '.NS' in ticker else 'USD',
            'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
            'high_52w': info.get('fiftyTwoWeekHigh', 'N/A'),
            'low_52w': info.get('fiftyTwoWeekLow', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A')
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_enhanced_stock_info(ticker, stock_db=None):
    """Get stock info with Neo4j enhancements"""
    basic_info = get_stock_info(ticker)
    
    if not basic_info or not stock_db:
        return basic_info
    
    # Add Neo4j enhancements
    try:
        correlated_stocks = stock_db.get_correlated_stocks(ticker, min_correlation=0.6)
        basic_info['correlated_stocks'] = correlated_stocks
    except Exception as e:
        logging.error(f"Error getting correlations for {ticker}: {e}")
        basic_info['correlated_stocks'] = []
    
    return basic_info

def create_advanced_chart(hist, ticker, company_name):
    """Create an advanced, beautiful stock chart"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist['Date'],
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name="OHLC",
        increasing=dict(line=dict(color="#00D4AA"), fillcolor="#00D4AA"),
        decreasing=dict(line=dict(color="#FF6B6B"), fillcolor="#FF6B6B")
    ))
    
    # Add moving averages
    if len(hist) >= 20:
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=hist['Date'],
            y=hist['MA20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='#4ECDC4', width=2, dash='dash')
        ))
    
    if len(hist) >= 50:
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=hist['Date'],
            y=hist['MA50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='#45B7D1', width=2, dash='dot')
        ))
    
    # Add volume as subplot
    fig.add_trace(go.Bar(
        x=hist['Date'],
        y=hist['Volume'],
        name="Volume",
        yaxis="y2",
        opacity=0.3,
        marker_color='#96CEB4'
    ))
    
    # Update layout for better aesthetics
    currency_symbol = "₹" if ".NS" in ticker else "$"
    
    fig.update_layout(
        title={
            'text': f"{company_name} ({ticker}) Stock Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        yaxis_title=f"Price ({currency_symbol})",
        xaxis_title="Date",
        template="plotly_white",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date"
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
    )
    
    # Add hover template
    fig.update_traces(
        hovertext=hist.apply(lambda row: f"Date: {row['Date'].date()}<br>"
                                        f"Open: {row['Open']:.2f}<br>"
                                        f"High: {row['High']:.2f}<br>"
                                        f"Low: {row['Low']:.2f}<br>"
                                        f"Close: {row['Close']:.2f}", axis=1),
        hoverinfo="text"
    )
    
    return fig

def show_correlation_analysis(stock_info, currency_symbol):
    """Display correlation analysis"""
    st.subheader("🔗 Market Relationships")
    
    correlated_stocks = stock_info.get('correlated_stocks', [])
    
    if correlated_stocks:
        # Create correlation chart
        tickers = [s['ticker'] for s in correlated_stocks[:5]]
        names = [s['name'][:20] for s in correlated_stocks[:5]]
        correlations = [s['correlation'] for s in correlated_stocks[:5]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=correlations,
                marker_color=['#27ae60' if c > 0.8 else '#f39c12' if c > 0.6 else '#e74c3c' for c in correlations],
                text=[f"{c:.2%}" for c in correlations],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Most Correlated Stocks",
            xaxis_title="Stock",
            yaxis_title="Correlation",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show details
        with st.expander("📋 Correlation Details"):
            for stock in correlated_stocks:
                correlation_pct = stock['correlation'] * 100
                strength = "Very Strong" if stock['correlation'] > 0.8 else "Strong" if stock['correlation'] > 0.6 else "Moderate"
                st.write(f"• **{stock['name']}** ({stock['ticker']}): {correlation_pct:.1f}% - {strength}")
    else:
        st.markdown("""
        <div class="correlation-container">
            <h4>💡 No correlation data available</h4>
            <p>Click 'Update Stock Relationships' in the sidebar to analyze market relationships.</p>
        </div>
        """, unsafe_allow_html=True)

def show_portfolio_analysis(watchlist):
    """Display portfolio analysis using Neo4j"""
    st.subheader("📊 Portfolio Intelligence")
    
    if len(watchlist) < 2:
        st.info("Add more stocks to see portfolio analysis")
        return
    
    try:
        # Get portfolio analysis from Neo4j
        risk_analysis = st.session_state.stock_db.get_portfolio_risk_analysis(watchlist)
        
        if risk_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sector Diversification**")
                sectors = [s['sector'] for s in risk_analysis]
                counts = [s['stock_count'] for s in risk_analysis]
                
                fig = go.Figure(data=[go.Pie(
                    labels=sectors,
                    values=counts,
                    hole=0.4
                )])
                fig.update_layout(
                    title="Portfolio by Sector",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Risk Assessment**")
                for sector_data in risk_analysis:
                    percentage = (sector_data['stock_count'] / len(watchlist)) * 100
                    risk_level = "High" if percentage > 50 else "Medium" if percentage > 30 else "Low"
                    risk_color = "#e74c3c" if percentage > 50 else "#f39c12" if percentage > 30 else "#27ae60"
                    
                    st.markdown(f"""
                    <div style="background: {risk_color}20; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid {risk_color}">
                        <strong>{sector_data['sector']}</strong><br>
                        {sector_data['stock_count']} stocks ({percentage:.1f}%) - {risk_level} concentration
                    </div>
                    """, unsafe_allow_html=True)
            
            # Diversification suggestions
            user_sectors = [s['sector'] for s in risk_analysis]
            suggestions = st.session_state.stock_db.find_diversification_opportunities(watchlist, user_sectors)
            
            if suggestions:
                st.write("**💡 Diversification Opportunities:**")
                for suggestion in suggestions:
                    st.write(f"• **{suggestion['sector']}**: {', '.join(suggestion['top_stocks'])}")
        else:
            st.info("Portfolio analysis data not available. Update stock relationships to see insights.")
    
    except Exception as e:
        st.error(f"Error analyzing portfolio: {str(e)}")

# --- FAQ Page ---
def faq_page():
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 StockChat </h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Ask me anything about the stock market!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not rag_available:
        st.error("❌ FAQ system is not available. Please check if the required dependencies are installed and the FAQ data file exists.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Question input
        user_question = st.text_area(
            "What would you like to know about the stock market?",
            placeholder="e.g., What is a dividend? How does the stock market work? What is P/E ratio?",
            height=100
        )
        
        if st.button("🔍 Get Answer", use_container_width=True):
            if user_question.strip():
                with st.spinner("🔍 Searching for the best answer..."):
                    result, similarity = find_best_answer(user_question, qa_pairs, embeddings, model)
                    
                    if result:
                        st.markdown(f"""
                        <div class="faq-container">
                            <div class="similarity-score">
                                📊 Confidence Score: {similarity:.2%}
                            </div>
                            <div class="faq-question">
                                ❓ Related Question: {result['question']}
                            </div>
                            <div class="faq-answer">
                                💡 <strong>Answer:</strong><br>
                                {result['answer']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-msg">
                            <h4>❌ No suitable answer found</h4>
                            <p>I couldn't find a good match for your question (confidence: {similarity:.2%}). 
                            Please try rephrasing your question or browse the available topics.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a question to get an answer.")
    
    with col2:
        st.subheader("📚 Available Topics")
        st.info("💡 Here are some topics I can help you with:")
        
        # Show available questions
        if qa_pairs:
            for i, qa in enumerate(qa_pairs[:10]):  # Show first 10 questions
                with st.expander(f"❓ {qa['question'][:50]}{'...' if len(qa['question']) > 50 else ''}"):
                    st.write(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")

# --- Login / Signup Page ---
def login_page():
    st.markdown("""
    <div class="main-header">
        <h1>🔐 Welcome to StockChat Market Explorer</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Your gateway to financial markets and investment tracking</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        auth_tabs = st.tabs(["🔑 Login", "👤 Sign Up"])
        
        with auth_tabs[0]:  # Login Tab
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                login_username = st.text_input("Username", placeholder="Enter your username")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if login_button:
                    if not login_username or not login_password:
                        st.markdown('<div class="error-msg">❌ Please enter both username and password.</div>', unsafe_allow_html=True)
                    else:
                        user_id = authenticate_user(login_username, login_password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.session_state.username = login_username
                            st.session_state.auth_success = True
                            st.markdown('<div class="success-msg">✅ Login successful! Redirecting...</div>', unsafe_allow_html=True)
                            st.rerun()
                        else:
                            st.markdown('<div class="error-msg">❌ Invalid credentials! Please try again.</div>', unsafe_allow_html=True)

        with auth_tabs[1]:  # Sign Up Tab
            st.subheader("Create New Account")
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Enter a unique username")
                new_password = st.text_input("Choose Password", type="password", placeholder="Enter a secure password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                signup_button = st.form_submit_button("🎯 Create Account", use_container_width=True)
                
                if signup_button:
                    if not all([new_username, new_password, confirm_password]):
                        st.markdown('<div class="error-msg">❌ Please fill all fields.</div>', unsafe_allow_html=True)
                    elif new_password != confirm_password:
                        st.markdown('<div class="error-msg">❌ Passwords do not match.</div>', unsafe_allow_html=True)
                    elif len(new_password) < 6:
                        st.markdown('<div class="error-msg">❌ Password must be at least 6 characters long.</div>', unsafe_allow_html=True)
                    else:
                        result = create_user(new_username, new_password)
                        if result:
                            st.markdown('<div class="success-msg">✅ Account created successfully! Please login now.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-msg">❌ Username already exists! Please choose another.</div>', unsafe_allow_html=True)

# --- Enhanced Dashboard Page ---
def enhanced_dashboard_page():
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 StockChat Market Explorer</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">Welcome back, <strong>{st.session_state.username}</strong>! Track your investments and explore the markets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Hello, {st.session_state.username}!")
        
        # Neo4j Connection Status
        if st.session_state.neo4j_connected:
            st.markdown('<div class="neo4j-status neo4j-connected">🔗 Graph DB Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="neo4j-status neo4j-disconnected">❌ Graph DB Offline</div>', unsafe_allow_html=True)
            if NEO4J_AVAILABLE:
                st.info("💡 Install and configure Neo4j for advanced analytics")
            else:
                st.info("💡 Install neo4j package: pip install neo4j")
        
        if st.button("🚪 Logout", use_container_width=True):
            # Clean up Neo4j connection
            if st.session_state.stock_db:
                st.session_state.stock_db.close()
            
            for key in ['user_id', 'username', 'auth_success', 'neo4j_connected', 'stock_db', 'analyzer']:
                if key in st.session_state:
                    st.session_state[key] = None if key not in ['auth_success', 'neo4j_connected'] else False
            st.rerun()
        
        st.markdown("---")
        
        # Stock Search
        st.subheader("📈 Stock Search")
        company_input = st.text_input("Enter Company Name or Ticker:", 
                                     placeholder="e.g., Apple, AAPL, Reliance")
        
        # Quick Select Popular Stocks
        st.subheader("🔥 Popular Stocks")
        popular_stock = st.selectbox("Quick Select:", 
                                   [""] + list(POPULAR_STOCKS.keys()))
        if popular_stock:
            company_input = popular_stock
        
        timeframe = st.selectbox("📅 Time Period:", 
                               ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                               index=5)
        
        specific_date = st.date_input("📍 Price on Date:", 
                                    value=datetime.today() - timedelta(days=1),
                                    max_value=datetime.today())
        
        # Neo4j-enhanced features
        if st.session_state.neo4j_connected:
            st.markdown("---")
            st.subheader("🔍 Market Intelligence")
            
            if company_input:
                ticker = get_ticker_from_input(company_input)
                if st.button("🔗 Update Stock Relationships", use_container_width=True):
                    with st.spinner("Analyzing market relationships..."):
                        try:
                            # Get watchlist for context
                            watchlist = get_watchlist(st.session_state.user_id)
                            all_tickers = list(set([ticker] + watchlist))
                            
                            updated_count = st.session_state.analyzer.update_stock_relationships(all_tickers)
                            st.success(f"✅ Updated {updated_count} stocks with relationships!")
                        except Exception as e:
                            st.error(f"Error updating relationships: {str(e)}")
        
        st.markdown("---")
        
        # Watchlist Management
        st.subheader("⭐ My Watchlist")
        
        # Add to watchlist
        with st.form("add_watchlist"):
            new_watch = st.text_input("Add Stock:", placeholder="Company name or ticker")
            add_button = st.form_submit_button("➕ Add to Watchlist")
            
            if add_button and new_watch:
                ticker = get_ticker_from_input(new_watch)
                # First check if the ticker is valid by trying to fetch data
                stock_info = get_stock_info(ticker)
                if stock_info:
                    if add_to_watchlist(st.session_state.user_id, ticker):
                        st.markdown('<div class="success-msg">✅ Added to watchlist successfully!</div>', unsafe_allow_html=True)
                        # Update Neo4j if connected
                        if st.session_state.neo4j_connected:
                            try:
                                st.session_state.analyzer.update_stock_relationships([ticker])
                            except Exception as e:
                                st.warning(f"Added to watchlist but couldn't update graph: {str(e)}")
                        st.rerun()
                    else:
                        st.markdown('<div class="error-msg">⚠️ Already in watchlist!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-msg">❌ Invalid ticker symbol!</div>', unsafe_allow_html=True)
        
        # Display enhanced watchlist
        watchlist = get_watchlist(st.session_state.user_id)
        if watchlist:
            for entry in watchlist:
                # Get enhanced info if Neo4j is connected
                if st.session_state.neo4j_connected:
                    stock_info = get_enhanced_stock_info(entry, st.session_state.stock_db)
                else:
                    stock_info = get_stock_info(entry)
                
                if stock_info:
                    currency_symbol = "₹" if stock_info['currency'] == 'INR' else "$"
                    change_class = "price-up" if stock_info['change'] >= 0 else "price-down"
                    change_symbol = "▲" if stock_info['change'] >= 0 else "▼"
                    
                    st.markdown(f"""
                    <div class="watchlist-container">
                        <div class="watchlist-header">
                            <div class="watchlist-name">{stock_info['name'][:25]}{'...' if len(stock_info['name']) > 25 else ''}</div>
                            <div class="watchlist-ticker">{entry}</div>
                        </div>
                        <div class="watchlist-price">{currency_symbol}{stock_info['price']:.2f}</div>
                        <div class="watchlist-change {change_class}">
                            {change_symbol} {stock_info['change_pct']:+.2f}% ({currency_symbol}{stock_info['change']:+.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show correlation info if available
                    if stock_info.get('correlated_stocks'):
                        corr_count = len(stock_info['correlated_stocks'])
                        st.caption(f"🔗 {corr_count} highly correlated stocks found")
                    
                    if st.button("🗑️ Remove", key=f"remove_{entry}", use_container_width=True):
                        if remove_from_watchlist(st.session_state.user_id, entry):
                            st.markdown('<div class="success-msg">✅ Removed from watchlist!</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.markdown(f"""
                    <div class="watchlist-container">
                        <div class="watchlist-name">{entry}</div>
                        <div style="color: #e74c3c;">Data unavailable</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑️ Remove", key=f"remove_{entry}", use_container_width=True):
                        if remove_from_watchlist(st.session_state.user_id, entry):
                            st.rerun()
        else:
            st.info("📝 Your watchlist is empty. Add some stocks to track!")

    # Main Content Area
    if company_input:
        ticker = get_ticker_from_input(company_input)
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=timeframe)
            
            if hist.empty:
                st.error(f"❌ No data found for '{company_input}' ({ticker}). Please check the ticker symbol.")
                return
            
            # Enhanced Stock Info Cards
            if st.session_state.neo4j_connected:
                stock_info = get_enhanced_stock_info(ticker, st.session_state.stock_db)
            else:
                stock_info = get_stock_info(ticker)
            
            if stock_info:
                col1, col2, col3, col4 = st.columns(4)
                
                currency_symbol = "₹" if stock_info['currency'] == 'INR' else "$"
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>💰 Current Price</h4>
                        <h2>{currency_symbol}{stock_info['price']:.2f}</h2>
                        <p style="color: {'#27ae60' if stock_info['change'] >= 0 else '#e74c3c'}">
                            {stock_info['change']:+.2f} ({stock_info['change_pct']:+.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📈 Period High</h4>
                        <h2>{currency_symbol}{hist['High'].max():.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📉 Period Low</h4>
                        <h2>{currency_symbol}{hist['Low'].min():.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    volume_str = f"{stock_info['volume']:,.0f}" if stock_info['volume'] != 0 else "N/A"
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📊 Volume</h4>
                        <h2>{volume_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced Price Chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader(f"📈 {stock_info['name'] if stock_info else ticker} Technical Analysis")
            
            hist.reset_index(inplace=True)
            
            # Create the advanced chart
            fig = create_advanced_chart(hist, ticker, stock_info['name'] if stock_info else ticker)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Neo4j-powered correlation analysis
            if st.session_state.neo4j_connected and stock_info.get('correlated_stocks'):
                show_correlation_analysis(stock_info, currency_symbol)
            
            # Price on specific date
            st.subheader(f"📅 Historical Price Lookup")
            date_str = specific_date.strftime("%Y-%m-%d")
            
            try:
                hist_date = stock.history(start=specific_date, end=specific_date + timedelta(days=1))
                if not hist_date.empty:
                    price_on_date = hist_date['Close'].iloc[0]
                    currency_symbol = "₹" if ".NS" in ticker else "$"
                    st.markdown(f"""
                    <div class="success-msg">
                        <h4>💰 Price on {date_str}</h4>
                        <h2>{currency_symbol}{price_on_date:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"📊 No trading data available for {date_str} (market may have been closed)")
            except:
                st.info(f"📊 Unable to fetch price data for {date_str}")
                
        except Exception as e:
            st.error(f"❌ Error fetching data for {company_input} ({ticker}): {str(e)}")
    
    else:
        # Market Overview with enhanced index cards
        st.subheader("🌍 Market Overview")
        
        indices = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN", 
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "FTSE 100": "^FTSE"
        }
        
        # Create two rows of 3 columns each
        rows = [list(indices.items())[i:i+3] for i in range(0, len(indices), 3)]
        
        for row in rows:
            cols = st.columns(3)
            for i, (name, ticker) in enumerate(row):
                with cols[i]:
                    try:
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="2d")
                        if not hist.empty:
                            current = hist['Close'].iloc[-1]
                            prev = hist['Close'].iloc[0] if len(hist) > 1 else current
                            change = current - prev
                            change_pct = (change / prev) * 100
                            
                            change_class = "positive-change" if change >= 0 else "negative-change"
                            change_symbol = "▲" if change >= 0 else "▼"
                            
                            st.markdown(f"""
                            <div class="index-card">
                                <div class="index-name">{name}</div>
                                <div class="index-price">{current:,.2f}</div>
                                <div class="index-change {change_class}">
                                    {change_symbol} {change_pct:+.2f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        st.markdown(f"""
                        <div class="index-card">
                            <div class="index-name">{name}</div>
                            <div class="index-price">N/A</div>
                            <div class="index-change">Data unavailable</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Neo4j-powered portfolio analysis
        if st.session_state.neo4j_connected:
            watchlist = get_watchlist(st.session_state.user_id)
            if watchlist and len(watchlist) >= 2:
                show_portfolio_analysis(watchlist)

# --- Main App Logic ---
if st.session_state.auth_success:
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 FAQ Assistant"])
    
    with tab1:
        enhanced_dashboard_page()
    
    with tab2:
        faq_page()
        
        # Handle auto-question from quick buttons
        if hasattr(st.session_state, 'auto_question'):
            if st.session_state.auto_question and rag_available:
                with st.spinner("🔍 Searching for the best answer..."):
                    result, similarity = find_best_answer(st.session_state.auto_question, qa_pairs, embeddings, model)
                    
                    if result:
                        st.markdown(f"""
                        <div class="faq-container">
                            <div class="similarity-score">
                                📊 Confidence Score: {similarity:.2%}
                            </div>
                            <div class="faq-question">
                                ❓ Question: {result['question']}
                            </div>
                            <div class="faq-answer">
                                💡 <strong>Answer:</strong><br>
                                {result['answer']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clear the auto question
                    del st.session_state.auto_question
else:
    login_page()

# Clean up on app shutdown
import atexit

def cleanup():
    if 'stock_db' in st.session_state and st.session_state.stock_db:
        try:
            st.session_state.stock_db.close()
        except:
            pass

atexit.register(cleanup)