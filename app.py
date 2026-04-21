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
        st.warning("Загрузи файл resumes_generated.txt")

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
    hr_core.USE_FILTERS = use_auto or use_manual
    hr_core.USE_RERANKING = use_reranking
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold

    # Запуск анализа
    with st.spinner("Загружаем резюме и анализируем кандидатов... (может занять 20–60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        
        # Сохраняем оригинальные резюме для статистики
        original_count = len(resumes)
        
        # Собираем статистику
        stats = {
            'invalid_name': 0,
            'invalid_city': 0,
            'no_age': 0,
            'no_experience': 0,
            'non_it': 0,
            'filter_age': 0,
            'filter_experience': 0,
            'filter_city': 0,
            'filter_position': 0,
            'filter_skills': 0,
            'passed_all': 0
        }
        
        # Детальный список отсеянных
        rejected_details = []
        
        # Флаг использования фильтров
        use_filters_flag = use_auto or use_manual
        
        # Анализируем каждое резюме
        for resume in resumes:
            reason = None
            category = None
            
            raw_name = str(resume.get('name', '')).strip()
            raw_city = str(resume.get('city', '')).strip()
            raw_position = str(resume.get('desired_position', '')).lower()
            raw_skills = str(resume.get('skills', '')).lower()
            
            # Извлекаем возраст
            age = None
            age_raw = resume.get('age')
            if age_raw:
                age_str = str(age_raw)
                age_match = re.search(r'(\d+)', age_str)
                if age_match:
                    age = int(age_match.group(1))
            
            # Извлекаем опыт
            experience = None
            exp_raw = resume.get('experience')
            if exp_raw:
                exp_str = str(exp_raw)
                exp_match = re.search(r'(\d+)', exp_str)
                if exp_match:
                    experience = int(exp_match.group(1))
            
            # ========== ВАЛИДАЦИЯ ==========
            # Проверка имени
            if not raw_name or len(raw_name) < 3:
                reason = f"Некорректное имя: '{raw_name[:30]}'"
                category = "invalid_name"
                stats['invalid_name'] += 1
            
            # Проверка возраста
            elif age is None or not (16 <= age <= 70):
                reason = f"Некорректный возраст: {resume.get('age', 'не указан')}"
                category = "no_age"
                stats['no_age'] += 1
            
            # Проверка опыта
            elif experience is None or not (0 <= experience <= 50):
                reason = f"Некорректный опыт: {resume.get('experience', 'не указан')}"
                category = "no_experience"
                stats['no_experience'] += 1
            
            # Проверка города
            elif not is_valid_russian_city(raw_city):
                reason = f"Неизвестный город: '{raw_city[:30]}'"
                category = "invalid_city"
                stats['invalid_city'] += 1
            
            else:
                # IT проверка
                is_it, it_reason = is_it_candidate(resume, threshold=it_threshold)
                if not is_it:
                    reason = f"Не IT-специальность: {it_reason}"
                    category = "non_it"
                    stats['non_it'] += 1
                else:
                    # ========== HR-ФИЛЬТРЫ ==========
                    hr_filter_reason = None
                    
                    if use_filters_flag:
                        # Фильтр по возрасту
                        if min_age and age < min_age:
                            hr_filter_reason = f"Возраст {age} < {min_age}"
                            category = "filter_age"
                            stats['filter_age'] += 1
                        elif max_age and age > max_age:
                            hr_filter_reason = f"Возраст {age} > {max_age}"
                            category = "filter_age"
                            stats['filter_age'] += 1
                        # Фильтр по опыту
                        elif min_exp and experience < min_exp:
                            hr_filter_reason = f"Опыт {experience} < {min_exp}"
                            category = "filter_experience"
                            stats['filter_experience'] += 1
                        # Фильтр по городу
                        elif city_filter and city_filter.strip():
                            if city_filter.lower() not in normalize_city(raw_city).lower():
                                hr_filter_reason = f"Город '{raw_city}' ≠ '{city_filter}'"
                                category = "filter_city"
                                stats['filter_city'] += 1
                        # Фильтр по должности
                        elif position_keywords and len(position_keywords) > 0:
                            if not any(kw.lower() in raw_position for kw in position_keywords):
                                hr_filter_reason = f"Должность не содержит ключевых слов"
                                category = "filter_position"
                                stats['filter_position'] += 1
                        # Фильтр по навыкам
                        elif skills_keywords and len(skills_keywords) > 0:
                            if not any(kw.lower() in raw_skills for kw in skills_keywords):
                                hr_filter_reason = f"Навыки не содержат ключевых слов"
                                category = "filter_skills"
                                stats['filter_skills'] += 1
                    
                    if hr_filter_reason:
                        reason = hr_filter_reason
                    else:
                        stats['passed_all'] += 1
            
            if reason:
                rejected_details.append({
                    'name': raw_name[:35],
                    'city': raw_city[:20],
                    'age': age if age else '?',
                    'experience': experience if experience else '?',
                    'reason': reason
                })
        
        # Запускаем ранжирование
        top_candidates = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=resumes,
            manual_filters=manual_filters,
            use_filters=use_filters_flag,
            use_reranking=use_reranking,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )

    # ====================== ВЫВОД СТАТИСТИКИ ======================
    st.success(f"✅ Готово! Найдено {len(top_candidates)} лучших кандидатов")
    
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Всего резюме", original_count)
    with col2:
        st.metric("✅ Прошло фильтры", stats['passed_all'])
    with col3:
        st.metric("❌ Отсеяно", original_count - stats['passed_all'])
    with col4:
        if original_count > 0:
            pass_rate = (stats['passed_all'] / original_count) * 100
            st.metric("📊 Процент отбора", f"{pass_rate:.1f}%")
    
    # Детальная статистика отсева
    with st.expander("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ОТСЕВА", expanded=True):
        
        st.subheader("Причины отсева кандидатов")
        
        reasons_data = []
        reason_names = {
            'invalid_name': '❌ Некорректное имя',
            'invalid_city': '🏙️ Неизвестный город',
            'no_age': '📅 Проблема с возрастом',
            'no_experience': '💼 Проблема с опытом',
            'non_it': '💻 Не IT-специальность',
            'filter_age': '🔞 Не прошёл возраст',
            'filter_experience': '📈 Не прошёл опыт',
            'filter_city': '📍 Не прошёл город',
            'filter_position': '💼 Должность не подходит',
            'filter_skills': '🛠️ Навыки не подходят'
        }
        
        for key, name in reason_names.items():
            count = stats.get(key, 0)
            if count > 0:
                reasons_data.append({'Причина': name, 'Количество': count})
        
        if reasons_data:
            df_reasons = pd.DataFrame(reasons_data)
            st.dataframe(df_reasons, use_container_width=True, hide_index=True)
            
            # Прогресс-бары
            for reason in reasons_data:
                percent = (reason['Количество'] / original_count) * 100
                st.progress(percent / 100, text=f"{reason['Причина']}: {reason['Количество']} ({percent:.1f}%)")
        else:
            st.info("✅ Все кандидаты прошли фильтры!")
        
        # Таблица отсеянных кандидатов
        if rejected_details:
            st.subheader(f"📋 Список отсеянных кандидатов ({len(rejected_details)})")
            df_rejected = pd.DataFrame(rejected_details)
            st.dataframe(df_rejected, use_container_width=True, hide_index=True)
    
    # Топ кандидатов
    if top_candidates:
        st.subheader("🏆 ТОП кандидатов")
        df = pd.DataFrame(top_candidates)
        cols_to_show = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary']
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        
        st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)
        
        # Детальная информация по топ-кандидатам
        with st.expander("📋 Детальная информация о топ-кандидатах"):
            for i, cand in enumerate(top_candidates[:5], 1):
                st.markdown(f"**{i}. {cand.get('name', 'Неизвестно')}**")
                st.text(f"   Возраст: {cand.get('parsed_age', '?')} лет")
                st.text(f"   Опыт: {cand.get('parsed_experience', '?')} лет")
                st.text(f"   Город: {cand.get('city', '?')}")
                st.text(f"   Зарплата: {cand.get('salary', '?')}")
                st.text(f"   Совместимость: {cand.get('score', 0)}%")
                if cand.get('rerank_score_percent'):
                    st.text(f"   Точность (rerank): {cand.get('rerank_score_percent')}%")
                st.markdown("---")

    else:
        st.warning("⚠️ Не найдено кандидатов, соответствующих критериям")
        
        with st.expander("💡 Как улучшить результаты?"):
            st.markdown("""
            **Попробуйте:**
            - Снизить минимальный опыт
            - Расширить возрастные рамки
            - Убрать или добавить город
            - Снизить порог IT Threshold
            - Уменьшить минимальный semantic score
            - Добавить больше ключевых слов в навыки
            """)

st.caption("Сделано на Streamlit + SentenceTransformer • Твоя HR-система 2026")