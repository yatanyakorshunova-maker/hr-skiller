import streamlit as st
import pandas as pd
import io
import sys
from typing import List, Dict

# ====================== ИМПОРТ ТВОИХ МОДУЛЕЙ ======================
import hr_core                     # ← твой основной скрипт
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

st.set_page_config(page_title="AI HR Скринер", layout="wide")
st.title("🚀 AI HR Скринер")
st.markdown("**Умный подбор Middle Backend Python-разработчиков**")

# ====================== Сайдбар — все фильтры ======================
with st.sidebar:
    st.header("⚙️ Настройки")
    
    use_auto = st.checkbox("Авто-фильтры из вакансии", value=True)
    use_manual = st.checkbox("Ручные фильтры", value=True)
    
    st.subheader("Требования к кандидату")
    min_age = st.slider("Возраст от", 18, 60, 23)
    max_age = st.slider("Возраст до", 18, 70, 45)
    min_exp = st.slider("Опыт от (лет)", 0, 15, 5)
    
    city_input = st.text_input("Город (пусто = любой)", value="")
    
    pos_kw = st.multiselect(
        "Должность содержит",
        ["Backend", "Python", "Developer", "Middle", "Senior"],
        default=["Backend", "Python", "Developer"]
    )
    skill_kw = st.multiselect(
        "Навыки содержат",
        ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL"],
        default=["Python", "FastAPI", "PostgreSQL", "Docker"]
    )
    
    it_threshold = st.slider("IT-порог", 0.0, 10.0, 2.0, 0.5)
    use_rerank = st.checkbox("Включить reranking", value=True)
    min_score = st.slider("Мин. semantic score", 10.0, 50.0, 15.0)
    top_k = st.slider("Топ кандидатов", 5, 50, 20)

# ====================== Основная область ======================
vacancy_text = st.text_area(
    "Текст вакансии",
    height=280,
    value="""Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой (готовы рассматривать релокацию)"""
)

uploaded = st.file_uploader("Файл с резюме (resumes_generated.txt)", type="txt")

if st.button("🔥 Запустить подбор", type="primary", use_container_width=True):
    # Сохраняем загруженный файл
    if uploaded:
        with open("resumes_generated.txt", "wb") as f:
            f.write(uploaded.getbuffer())
        resume_file = "resumes_generated.txt"
        st.success("Резюме загружено")
    else:
        resume_file = "resumes_generated.txt"

    # Подготовка ручных фильтров
    manual_filters = {
        'min_age': min_age,
        'max_age': max_age,
        'min_experience': min_exp,
        'city': city_input.strip() if city_input.strip() else None,
        'position_keywords': pos_kw,
        'skills_keywords': skill_kw
    }

    # Устанавливаем настройки в hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(city_input.strip())
    hr_core.USE_POSITION_KEYWORDS = bool(pos_kw)
    hr_core.USE_SKILLS_KEYWORDS = bool(skill_kw)
    hr_core.USE_FILTERS = True
    hr_core.USE_RERANKING = use_rerank
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold

    # Загрузка резюме
    with st.spinner("Загружаем резюме..."):
        resumes = hr_core.load_resumes_from_file(resume_file)

    # Захватываем весь консольный вывод
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    with st.spinner("ИИ анализирует кандидатов (10–40 сек)..."):
        top_candidates = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=resumes,
            manual_filters=manual_filters,
            use_filters=True,
            use_reranking=use_rerank,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )

    sys.stdout = old_stdout
    console_log = captured.getvalue()

    # ====================== Результаты ======================
    st.success(f"✅ Найдено {len(top_candidates)} лучших кандидатов")

    if top_candidates:
        df = pd.DataFrame(top_candidates)
        display_cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary']
        display_cols = [c for c in display_cols if c in df.columns]
        
        st.subheader("🏆 ТОП кандидатов")
        st.dataframe(
            df[display_cols].rename(columns={
                'name': 'Имя',
                'parsed_age': 'Возраст',
                'parsed_experience': 'Опыт',
                'score': 'Score %',
                'rerank_score_percent': 'Rerank %'
            }),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("📋 Полный лог + статистика отсева"):
        st.text(console_log)

    st.balloons()

st.caption("Streamlit + SentenceTransformer + твой AI-скринер • 2026")