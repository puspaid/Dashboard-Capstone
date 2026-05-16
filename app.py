import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #0a0f1e; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2035, #1e2a45);
        border: 1px solid #2d3f6e;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .label {
        color: #8899bb;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-card .icon {
        font-size: 20px;
        margin-bottom: 4px;
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 22px;
        font-weight: 700;
        padding: 10px 0 4px 0;
        border-bottom: 2px solid #2d3f6e;
        margin-bottom: 16px;
    }

    /* Sidebar title */
    .sidebar-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        padding: 10px 0;
    }
    .sidebar-subtitle {
        color: #8899bb;
        font-size: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .group-name {
        color: #f97316;
        font-size: 13px;
        text-align: center;
        font-weight: 600;
        margin-top: 4px;
    }
    .project-name {
        color: #60a5fa;
        font-size: 11px;
        text-align: center;
        margin-bottom: 16px;
    }

    /* Page title */
    .page-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #8899bb;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #1a2035, #1e2a45);
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #d1d5db;
        font-size: 14px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Plot background */
    .stPlot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data():
    url = 'https://raw.githubusercontent.com/fahlevirafly29/Capstone-Project/main/Dataset_After_Wranglingg.zip'
    df = pd.read_csv(url, compression='zip')
    df['summary_wlen'] = df['summary'].fillna('').astype(str).str.split().str.len()
    df['exp_wlen'] = df['experience_desc'].fillna('').astype(str).str.split().str.len()
    df['is_senior'] = df['years_experience'] > 10
    return df

# ── Plot styling helper ──────────────────────────────────────────────────────
def dark_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a2035')
    ax.tick_params(colors='#8899bb')
    ax.xaxis.label.set_color('#8899bb')
    ax.yaxis.label.set_color('#8899bb')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3f6e')
    return fig, ax

def dark_fig_multi(rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor('#0e1117')
    ax_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in ax_flat:
        ax.set_facecolor('#1a2035')
        ax.tick_params(colors='#8899bb')
        ax.xaxis.label.set_color('#8899bb')
        ax.yaxis.label.set_color('#8899bb')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3f6e')
    return fig, axes

PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
JOB_LABELS = {'software_engineer': 'Software Engineer', 'data_analyst': 'Data Analyst',
               'teacher': 'Teacher', 'admin': 'Admin'}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:48px;'>🎯</div>
        <div class='sidebar-title'>CV Analytics</div>
        <div class='sidebar-subtitle'>Online Analytics Platform</div>
        <div class='group-name'>Ready To Perform</div>
        <div class='project-name'>Capstone Project CC26-PSU078</div>
    </div>
    <hr style='border-color:#2d3f6e; margin: 8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Filter Data")

    # Load data first (needed for filter options)
    with st.spinner("Memuat dataset..."):
        df_raw = load_data()

    all_jobs = df_raw['job'].unique().tolist()
    selected_jobs = st.multiselect(
        "Pilih Profesi",
        options=all_jobs,
        default=all_jobs,
        format_func=lambda x: JOB_LABELS.get(x, x)
    )

    min_exp = int(df_raw['years_experience'].min())
    max_exp = int(df_raw['years_experience'].max())
    exp_range = st.slider(
        "Rentang Pengalaman (Tahun)",
        min_value=min_exp, max_value=max_exp,
        value=(min_exp, max_exp)
    )

    senior_only = st.checkbox("Tampilkan Kandidat Senior Saja (>10 Tahun)", value=False)

    st.markdown("<hr style='border-color:#2d3f6e; margin: 16px 0;'>", unsafe_allow_html=True)

    # Page navigation
    st.markdown("### 🗂️ Main Menu")
    page = st.radio(
        "",
        ["🏠 Home", "📊 Kompetensi (PB-1)", "📈 Pengalaman (PB-2)", "🔤 Teks NLP (PB-3)", "📋 Kesimpulan"],
        label_visibility="collapsed"
    )

# ── Filter df ────────────────────────────────────────────────────────────────
df = df_raw[
    (df_raw['job'].isin(selected_jobs)) &
    (df_raw['years_experience'] >= exp_range[0]) &
    (df_raw['years_experience'] <= exp_range[1])
].copy()
if senior_only:
    df = df[df['is_senior']]

if df.empty:
    st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Coba sesuaikan filter Anda.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='page-title'>🎯 CV Analytics Dashboard</div>
    <div class='page-subtitle'>Capstone Project CC26-PSU078 — Kelompok <b style='color:#f97316'>Ready To Perform</b></div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    total = len(df)
    avg_exp = df['years_experience'].mean()
    senior_pct = (df['is_senior'].sum() / total * 100) if total > 0 else 0
    most_common_job = df['job'].mode()[0] if total > 0 else "-"
    most_common_label = JOB_LABELS.get(most_common_job, most_common_job)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "👥", "Total Kandidat", f"{total:,}"),
        (c2, "📅", "Rata-rata Pengalaman", f"{avg_exp:.1f} Thn"),
        (c3, "🏆", "Kandidat Senior", f"{senior_pct:.1f}%"),
        (c4, "💼", "Profesi Terbanyak", most_common_label),
        (c5, "📑", "Jumlah Profesi", str(len(df['job'].unique()))),
    ]
    for col, icon, label, value in cards:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='icon'>{icon}</div>
            <div class='label'>{label}</div>
            <div class='value'>{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Distribution by Job + Experience Boxplot ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>📊 Distribusi Kandidat per Profesi</div>", unsafe_allow_html=True)
        job_counts = df['job'].value_counts().reset_index()
        job_counts.columns = ['job', 'count']
        job_counts['label'] = job_counts['job'].map(JOB_LABELS)
        fig, ax = dark_fig(figsize=(7, 4))
        bars = ax.barh(job_counts['label'], job_counts['count'], color=PALETTE[:len(job_counts)], edgecolor='none', height=0.55)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + total * 0.005, bar.get_y() + bar.get_height()/2,
                    f'{int(w):,}', va='center', color='#ffffff', fontsize=10, fontweight='600')
        ax.set_xlabel('Jumlah Kandidat')
        ax.set_xlim(0, job_counts['count'].max() * 1.15)
        ax.set_title('Jumlah Kandidat per Profesi', color='#ffffff', fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("<div class='section-header'>📈 Distribusi Pengalaman Kerja</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(7, 4))
        job_order = df['job'].unique().tolist()
        palette_map = {j: PALETTE[i % len(PALETTE)] for i, j in enumerate(job_order)}
        sns.boxplot(data=df, x='years_experience', y='job', palette=palette_map,
                    order=job_order, ax=ax, linewidth=1.2, flierprops=dict(marker='o', color='#f97316', markersize=3))
        ax.axvline(x=10, color='#f97316', linestyle='--', linewidth=1.5, label='Batas Senior (>10 Thn)')
        ax.set_yticklabels([JOB_LABELS.get(j, j) for j in job_order], color='#d1d5db')
        ax.set_xlabel('Tahun Pengalaman Kerja')
        ax.set_ylabel('')
        ax.set_title('Distribusi Pengalaman per Profesi', color='#ffffff', fontweight='bold')
        ax.legend(facecolor='#1a2035', edgecolor='#2d3f6e', labelcolor='#d1d5db', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2: Summary word length distribution ──
    st.markdown("<div class='section-header'>📝 Panjang Teks CV per Profesi</div>", unsafe_allow_html=True)
    fig, axes = dark_fig_multi(1, 2, figsize=(14, 4))
    axes = axes.flatten()
    for i, (col, title) in enumerate([('summary_wlen', 'Kolom Summary'), ('exp_wlen', 'Kolom Experience Desc')]):
        for j, job in enumerate(df['job'].unique()):
            subset = df[df['job'] == job][col]
            if len(subset) > 1:
                axes[i].fill_between([], [], alpha=0.35)  # placeholder
                axes[i].hist(subset, bins=30, alpha=0.45, label=JOB_LABELS.get(job, job),
                             color=PALETTE[j % len(PALETTE)], edgecolor='none')
        axes[i].set_title(f'Distribusi Panjang Kata — {title}', color='#ffffff', fontweight='bold')
        axes[i].set_xlabel('Jumlah Kata')
        axes[i].set_ylabel('Frekuensi')
        axes[i].legend(facecolor='#1a2035', edgecolor='#2d3f6e', labelcolor='#d1d5db', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE PB-1: KOMPETENSI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Kompetensi (PB-1)":
    st.markdown("""
    <div class='page-title'>📊 Profil Kompetensi (PB-1)</div>
    <div class='page-subtitle'>Ekstraksi kata kunci dan frasa dominan dari teks CV kandidat per profesi</div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>☁️ WordCloud — Kolom Summary</div>", unsafe_allow_html=True)
    jobs_available = df['job'].unique().tolist()
    n_jobs = len(jobs_available)
    if n_jobs == 0:
        st.warning("Tidak ada data.")
        st.stop()

    cols_wc = st.columns(min(n_jobs, 4))
    cmaps = ['Blues', 'Greens', 'Oranges', 'Reds']
    for i, job in enumerate(jobs_available[:4]):
        with cols_wc[i]:
            teks = ' '.join(df[df['job'] == job]['summary'].dropna().astype(str))
            if len(teks.strip()) < 10:
                st.info(f"Teks tidak cukup untuk {JOB_LABELS.get(job, job)}")
                continue
            wc = WordCloud(width=350, height=280, background_color='#1a2035',
                           colormap=cmaps[i % len(cmaps)], max_words=80).generate(teks)
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#1a2035')
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(JOB_LABELS.get(job, job), fontweight='bold', color='#ffffff', fontsize=11)
            ax.axis('off')
            fig.tight_layout(pad=0.3)
            st.pyplot(fig)
            plt.close()

    st.markdown("<br><div class='section-header'>📊 Top-5 Bi-gram — Experience Description</div>", unsafe_allow_html=True)

    def get_top_bigrams(series, top_n=5):
        series = series.dropna().astype(str)
        if len(series) < 2:
            return []
        try:
            vec = CountVectorizer(ngram_range=(2,2), stop_words='english', max_features=200).fit(series)
            bow = vec.transform(series)
            sum_words = bow.sum(axis=0)
            words_freq = [(w, sum_words[0, idx]) for w, idx in vec.vocabulary_.items()]
            return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
        except:
            return []

    n_cols = min(n_jobs, 2)
    n_rows = (n_jobs + 1) // 2
    fig, axes = dark_fig_multi(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, job in enumerate(jobs_available):
        top_bg = get_top_bigrams(df[df['job'] == job]['experience_desc'])
        ax = axes_flat[i]
        if not top_bg:
            ax.text(0.5, 0.5, 'Data tidak cukup', ha='center', va='center', color='#8899bb')
            continue
        frasa, frek = zip(*top_bg)
        color = PALETTE[i % len(PALETTE)]
        bars = ax.barh(list(frasa)[::-1], list(frek)[::-1], color=color, edgecolor='none', height=0.55)
        for bar, val in zip(bars, list(frek)[::-1]):
            ax.text(bar.get_width() + max(frek) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(val):,}', va='center', color='#ffffff', fontsize=10)
        ax.set_title(f'Top Bi-gram: {JOB_LABELS.get(job, job)}', color='#ffffff', fontweight='bold')
        ax.set_xlabel('Frekuensi Kemunculan')
        ax.set_xlim(0, max(frek) * 1.18)
        ax.tick_params(axis='y', labelsize=9, colors='#d1d5db')
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Insight PB-1:</b> Setiap profesi memiliki profil kompetensi yang sangat khas dan tidak saling tumpang tindih.
    Ini menjadi sinyal kuat bahwa model NLP kita akan mampu membedakan profesi pelamar berdasarkan teks CV mereka.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE PB-2: PENGALAMAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Pengalaman (PB-2)":
    st.markdown("""
    <div class='page-title'>📈 Distribusi Pengalaman Kerja (PB-2)</div>
    <div class='page-subtitle'>Analisis rata-rata pengalaman kerja dan distribusi kandidat senior per profesi</div>
    """, unsafe_allow_html=True)

    # KPI row
    rata_rata = df.groupby('job')['years_experience'].mean()
    jobs_available = df['job'].unique().tolist()
    cols_kpi = st.columns(len(jobs_available))
    for i, job in enumerate(jobs_available):
        avg = rata_rata.get(job, 0)
        cols_kpi[i].markdown(f"""
        <div class='metric-card'>
            <div class='icon'>📅</div>
            <div class='label'>{JOB_LABELS.get(job, job)}</div>
            <div class='value'>{avg:.1f} Thn</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>📦 Boxplot Pengalaman per Profesi</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(7, 5))
        palette_map = {j: PALETTE[i % len(PALETTE)] for i, j in enumerate(jobs_available)}
        sns.boxplot(data=df, x='years_experience', y='job', palette=palette_map,
                    order=jobs_available, ax=ax, linewidth=1.2,
                    flierprops=dict(marker='o', color='#f97316', markersize=3))
        ax.axvline(x=10, color='#f97316', linestyle='--', linewidth=1.8, label='Batas Senior (>10 Thn)')
        ax.set_yticklabels([JOB_LABELS.get(j, j) for j in jobs_available], color='#d1d5db')
        ax.set_xlabel('Tahun Pengalaman Kerja')
        ax.set_ylabel('')
        ax.set_title('Distribusi Pengalaman per Profesi', color='#ffffff', fontweight='bold')
        ax.legend(facecolor='#1a2035', edgecolor='#2d3f6e', labelcolor='#d1d5db')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("<div class='section-header'>🏆 Jumlah Kandidat Senior (>10 Tahun)</div>", unsafe_allow_html=True)
        senior_counts = df[df['is_senior']].groupby('job').size().reset_index(name='count')
        if senior_counts.empty:
            st.info("Tidak ada kandidat senior dalam filter ini.")
        else:
            senior_counts = senior_counts.sort_values('count', ascending=False)
            fig, ax = dark_fig(figsize=(7, 5))
            colors_s = [PALETTE[jobs_available.index(j) % len(PALETTE)] if j in jobs_available else PALETTE[0]
                        for j in senior_counts['job']]
            bars = ax.barh(
                [JOB_LABELS.get(j, j) for j in senior_counts['job']],
                senior_counts['count'], color=colors_s, edgecolor='none', height=0.55
            )
            for bar in bars:
                w = bar.get_width()
                ax.text(w + senior_counts['count'].max() * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{int(w):,}', va='center', color='#ffffff', fontsize=11, fontweight='600')
            ax.set_xlabel('Jumlah Kandidat Senior')
            ax.set_xlim(0, senior_counts['count'].max() * 1.18)
            ax.set_title('Kandidat Senior per Profesi', color='#ffffff', fontweight='bold')
            ax.tick_params(axis='y', colors='#d1d5db')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Rata-rata bar
    st.markdown("<div class='section-header'>📊 Rata-rata Pengalaman per Profesi</div>", unsafe_allow_html=True)
    fig, ax = dark_fig(figsize=(12, 4))
    rata_df = df.groupby('job')['years_experience'].mean().reset_index()
    rata_df = rata_df.sort_values('years_experience', ascending=False)
    colors_r = [PALETTE[jobs_available.index(j) % len(PALETTE)] if j in jobs_available else PALETTE[0]
                for j in rata_df['job']]
    bars = ax.bar([JOB_LABELS.get(j, j) for j in rata_df['job']],
                  rata_df['years_experience'], color=colors_r, edgecolor='none', width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', color='#ffffff', fontsize=11)
    ax.set_ylabel('Rata-rata (Tahun)')
    ax.set_title('Rata-rata Pengalaman Kerja per Profesi', color='#ffffff', fontweight='bold')
    ax.tick_params(axis='x', colors='#d1d5db')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Insight PB-2:</b> <b>Software Engineer</b> merupakan jabatan yang paling banyak dihuni oleh kandidat senior (>10 thn).
    Sebaliknya, posisi <b>Admin</b> lebih banyak diisi oleh kandidat level pemula dengan rata-rata pengalaman pendek.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE PB-3: TEKS NLP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 Teks NLP (PB-3)":
    st.markdown("""
    <div class='page-title'>🔤 Pola Karakteristik Teks NLP (PB-3)</div>
    <div class='page-subtitle'>Analisis distribusi panjang kata dan bi-gram sebagai persiapan pemodelan NLP</div>
    """, unsafe_allow_html=True)

    jobs_available = df['job'].unique().tolist()

    # KDE Distribution
    st.markdown("<div class='section-header'>📉 Distribusi Panjang Kata per Profesi (KDE)</div>", unsafe_allow_html=True)
    fig, axes = dark_fig_multi(1, 2, figsize=(14, 5))
    axes = axes.flatten()
    config = [('summary_wlen', 'Kolom `summary`'), ('exp_wlen', 'Kolom `experience_desc`')]
    for i, (col, title) in enumerate(config):
        for j, job in enumerate(jobs_available):
            subset = df[df['job'] == job][col]
            if len(subset) > 1:
                sns.kdeplot(subset, ax=axes[i], label=JOB_LABELS.get(job, job),
                            fill=True, alpha=0.35, color=PALETTE[j % len(PALETTE)], linewidth=1.8)
        axes[i].set_title(f'Distribusi Panjang Kata — {title}', color='#ffffff', fontweight='bold')
        axes[i].set_xlabel('Jumlah Kata')
        axes[i].set_ylabel('Kepadatan (Density)')
        axes[i].legend(facecolor='#1a2035', edgecolor='#2d3f6e', labelcolor='#d1d5db', fontsize=9)
        axes[i].tick_params(colors='#8899bb')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tabel statistik
    st.markdown("<div class='section-header'>📋 Statistik Deskriptif Panjang Teks</div>", unsafe_allow_html=True)
    stats_df = df.groupby('job').agg(
        mean_summary=('summary_wlen', 'mean'),
        median_summary=('summary_wlen', 'median'),
        mean_exp=('exp_wlen', 'mean'),
        median_exp=('exp_wlen', 'median'),
    ).reset_index()
    stats_df['job'] = stats_df['job'].map(JOB_LABELS)
    stats_df.columns = ['Profesi', 'Rata-rata Summary', 'Median Summary', 'Rata-rata Exp Desc', 'Median Exp Desc']
    stats_df = stats_df.round(1)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Top Bigram Summary
    st.markdown("<div class='section-header'>📊 Top-5 Bi-gram Summary per Profesi</div>", unsafe_allow_html=True)

    def get_top_bigrams(series, top_n=5):
        series = series.dropna().astype(str)
        if len(series) < 2:
            return []
        try:
            vec = CountVectorizer(ngram_range=(2,2), stop_words='english', max_features=200).fit(series)
            bow = vec.transform(series)
            sum_words = bow.sum(axis=0)
            words_freq = [(w, sum_words[0, idx]) for w, idx in vec.vocabulary_.items()]
            return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
        except:
            return []

    n_jobs = len(jobs_available)
    n_cols = min(n_jobs, 2)
    n_rows = (n_jobs + 1) // 2
    fig, axes = dark_fig_multi(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, job in enumerate(jobs_available):
        top_bg = get_top_bigrams(df[df['job'] == job]['summary'])
        ax = axes_flat[i]
        if not top_bg:
            ax.text(0.5, 0.5, 'Data tidak cukup', ha='center', va='center', color='#8899bb')
            continue
        frasa, frek = zip(*top_bg)
        color = PALETTE[i % len(PALETTE)]
        bars = ax.barh(list(frasa)[::-1], list(frek)[::-1], color=color, edgecolor='none', height=0.5)
        for bar, val in zip(bars, list(frek)[::-1]):
            ax.text(bar.get_width() + max(frek)*0.01, bar.get_y()+bar.get_height()/2,
                    f'{int(val):,}', va='center', color='#ffffff', fontsize=10)
        ax.set_title(f'{JOB_LABELS.get(job, job)}', color='#ffffff', fontweight='bold')
        ax.set_xlabel('Frekuensi Kemunculan')
        ax.set_xlim(0, max(frek)*1.18)
        ax.tick_params(axis='y', labelsize=9, colors='#d1d5db')
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Insight PB-3:</b> Frasa dominan antar profesi tidak saling tumpang tindih — ini bukti empiris bahwa kolom <code>summary</code>
    mengandung sinyal diskriminatif yang kuat. Model TF-IDF atau embedding akan sangat efektif karena distribusi bi-gram tiap kelas
    sudah <em>linearly separable</em> secara semantik.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: KESIMPULAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Kesimpulan":
    st.markdown("""
    <div class='page-title'>📋 Kesimpulan & Rekomendasi</div>
    <div class='page-subtitle'>Ringkasan temuan EDA & ExDA untuk Capstone Project CC26-PSU078</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    🎯 <b>Tujuan Proyek:</b> Membangun produk berbasis AI untuk memprediksi profesi dari CV, membantu pencari kerja
    mencocokkan kompetensi mereka dan memudahkan perusahaan mendapatkan kandidat yang sesuai kriteria.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg,#1a2035,#1e2a45); border-radius:12px; padding:20px; margin-bottom:16px; border:1px solid #2d3f6e;'>
        <h4 style='color:#3b82f6; margin:0 0 12px 0;'>✅ PB-1: Profil Kompetensi</h4>
        <p style='color:#d1d5db; font-size:14px;'>Setiap profesi memiliki kosakata yang sangat khas:</p>
        <ul style='color:#d1d5db; font-size:13px; line-height:1.8;'>
            <li><b style='color:#3b82f6;'>Software Engineer:</b> web development, machine learning</li>
            <li><b style='color:#10b981;'>Data Analyst:</b> power bi, sql server</li>
            <li><b style='color:#f59e0b;'>Teacher:</b> special education, lesson plans</li>
            <li><b style='color:#ef4444;'>Admin:</b> project management, customer service</li>
        </ul>
        <p style='color:#10b981; font-size:13px; font-weight:600;'>→ Data teks sangat ideal untuk NLP Classifier</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: linear-gradient(135deg,#1a2035,#1e2a45); border-radius:12px; padding:20px; border:1px solid #2d3f6e;'>
        <h4 style='color:#10b981; margin:0 0 12px 0;'>✅ PB-3: Karakteristik Teks NLP</h4>
        <p style='color:#d1d5db; font-size:13px; line-height:1.8;'>
        Dua properti ideal untuk pemodelan NLP:<br>
        <b>1. Panjang teks homogen</b> — memudahkan tokenisasi & padding. Rekomendasi <code>max_length</code> ≈ 45–50 kata (persentil ke-95).<br>
        <b>2. Bi-gram khas per profesi</b> — membuktikan data mengandung sinyal kuat untuk klasifikasi multi-kelas.<br>
        </p>
        <p style='color:#10b981; font-size:13px; font-weight:600;'>→ Dataset 100% siap untuk pemodelan</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg,#1a2035,#1e2a45); border-radius:12px; padding:20px; margin-bottom:16px; border:1px solid #2d3f6e;'>
        <h4 style='color:#f59e0b; margin:0 0 12px 0;'>✅ PB-2: Distribusi Pengalaman</h4>
        <p style='color:#d1d5db; font-size:14px;'>Temuan kunci:</p>
        <ul style='color:#d1d5db; font-size:13px; line-height:1.8;'>
            <li><b style='color:#3b82f6;'>Software Engineer</b> — paling banyak kandidat senior</li>
            <li><b style='color:#10b981;'>Teacher</b> — juga didominasi senior berpengalaman</li>
            <li><b style='color:#ef4444;'>Admin</b> — paling banyak entry-level, rata-rata pengalaman terendah</li>
        </ul>
        <p style='color:#10b981; font-size:13px; font-weight:600;'>→ Fitur numerik years_experience sangat diskriminatif</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: linear-gradient(135deg,#1e1a35,#2a1e45); border-radius:12px; padding:20px; border:1px solid #4d3f6e;'>
        <h4 style='color:#f97316; margin:0 0 12px 0;'>🚀 Rekomendasi untuk Tim AI</h4>
        <p style='color:#d1d5db; font-size:13px; line-height:1.8;'>
        Bangun arsitektur <b>Deep Learning multi-input</b>:<br><br>
        <b>📝 Input Teks:</b> Tokenization/Embedding pada kolom <code>summary</code> dan <code>experience_desc</code><br>
        <b>🔢 Input Numerik:</b> Nilai <code>years_experience</code> sebagai bobot tambahan<br><br>
        Perpaduan kedua fitur ini akan menghasilkan sistem <em>Automated Candidate Profiling</em> dengan akurasi tinggi.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Dataset preview
    st.markdown("<br><div class='section-header'>🗃️ Preview Dataset</div>", unsafe_allow_html=True)
    preview_cols = [c for c in ['job', 'years_experience', 'summary', 'experience_desc', 'is_senior'] if c in df.columns]
    st.dataframe(df[preview_cols].head(20), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style='text-align:center; padding:20px 0; color:#8899bb; font-size:13px;'>
    📊 Total data yang ditampilkan: <b style='color:#ffffff;'>{len(df):,} baris</b> |
    Kelompok <b style='color:#f97316;'>Ready To Perform</b> — Capstone Project CC26-PSU078
    </div>
    """, unsafe_allow_html=True)
