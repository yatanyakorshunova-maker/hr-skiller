import streamlit as st
import pandas as pd
import torch
import re
import warnings
import numpy as np
from scipy.special import softmax
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')

# ====================== ИМПОРТ ТВОИХ МОДУЛЕЙ ======================
import hr_core
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

st.set_page_config(page_title="AI HR Скринер", layout="wide")
st.title("🚀 AI HR Скринер — Подбор Python Backend разработчиков")
st.markdown("Умный подбор кандидатов с фильтрами и нейросетями")

# ====================== Сайдбар с фильтрами ======================
with st.sidebar:
    st.header("⚙️ Настройки фильтров")
    
    use_auto = st.checkbox("Использовать авто-фильтры из вакансии", value=True)
    use_manual = st.checkbox("Использовать ручные фильтры", value=True)
    
    st.subheader("Требования")
    min_age = st.slider("Минимальный возраст", 18, 60, 23)
    max_age = st.slider("Максимальный возраст", 20, 70, 45)
    min_exp = st.slider("Минимальный опыт (лет)", 0, 15, 4)
    
    city_filter = st.text_input("Город (оставь пустым = любой)", "")
    
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack"],
        default=["Backend", "Python", "Developer"]
    )
    
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL"],
        default=["Python", "FastAPI", "PostgreSQL", "Docker"]
    )
    
    it_threshold = st.slider("IT Threshold (чувствительность)", 0.0, 10.0, 2.0, 0.5)
    use_reranking = st.checkbox("Включить reranking (более точный)", value=True)
    min_score = st.slider("Минимальный semantic score (%)", 10, 40, 15)
    top_k = st.slider("Сколько кандидатов показывать", 5, 50, 20)

# ====================== Основная часть ======================
vacancy_text = st.text_area(
    "📝 Вставь текст вакансии сюда",
    height=250,
    value="""Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой"""
)

uploaded_file = st.file_uploader("Загрузи файл с резюме (resumes_generated.txt)", type=["txt"])

if st.button("🚀 Запустить подбор кандидатов", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with open("resumes_generated.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Файл с резюме загружен!")
    else:
        st.warning("Загрузи файл resumes_generated.txt или убедись, что он лежит в папке")

    # Подготовка ручных фильтров
    manual_filters = {
        'min_age': min_age,
        'max_age': max_age,
        'min_experience': min_exp,
        'city': city_filter.strip() if city_filter.strip() else None,
        'position_keywords': position_keywords,
        'skills_keywords': skills_keywords
    }

    # Применяем настройки к hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(city_filter.strip())
    hr_core.USE_POSITION_KEYWORDS = bool(position_keywords)
    hr_core.USE_SKILLS_KEYWORDS = bool(skills_keywords)
    hr_core.USE_FILTERS = True
    hr_core.USE_RERANKING = use_reranking
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold

    # Запуск анализа
    with st.spinner("Загружаем резюме и анализируем кандидатов... (может занять 20–60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        
        top_candidates = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=resumes,
            manual_filters=manual_filters,
            use_filters=True,
            use_reranking=use_reranking,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )

    st.success(f"✅ Готово! Найдено {len(top_candidates)} лучших кандидатов")

    if top_candidates:
        df = pd.DataFrame(top_candidates)
        cols_to_show = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary']
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        
        st.subheader("🏆 ТОП кандидатов")
        st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)

    with st.expander("📋 Полный лог работы (статистика отсева)"):
        # Здесь можно улучшить вывод лога позже
        st.info("Лог выводится в консоль. Запусти в терминале для полного просмотра.")

st.caption("Сделано на Streamlit + SentenceTransformer • Твоя HR-система 2026")