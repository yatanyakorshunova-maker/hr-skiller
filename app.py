import streamlit as st
import pandas as pd
import io
import sys
from typing import List, Dict

# ====================== ИМПОРТ ТВОИХ МОДУЛЕЙ ======================
import hr_core
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

# Подавляем вывод print из hr_core
import builtins
original_print = builtins.print

def silenced_print(*args, **kwargs):
    # Фильтруем сообщения от hr_core
    msg = str(args[0]) if args else ""
    skip_phrases = [
        "Выполняется семантическое ранжирование",
        "Запускаем reranking",
        "Автоматическое извлечение фильтров",
        "Применяются ручные фильтры",
        "Применяемые фильтры",
        "Начинаем валидацию и фильтрацию",
        "semantic search",
        "reranking"
    ]
    if any(phrase.lower() in msg.lower() for phrase in skip_phrases):
        return
    original_print(*args, **kwargs)

builtins.print = silenced_print

st.set_page_config(page_title="AI HR Скринер", layout="wide")
st.title("🚀 AI HR Скринер")

# ====================== Сайдбар — все фильтры с клавиатуры ======================
with st.sidebar:
    st.header("⚙️ Настройки фильтрации")
    
    st.subheader("👤 Возраст кандидата")
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.number_input("От (лет)", min_value=18, max_value=70, value=23, step=1)
    with col2:
        max_age = st.number_input("До (лет)", min_value=18, max_value=70, value=45, step=1)
    
    st.subheader("💼 Опыт работы")
    min_exp = st.number_input("Минимальный опыт (лет)", min_value=0, max_value=20, value=5, step=1)
    
    st.subheader("📍 Город")
    city_input = st.text_input("Город (оставьте пустым - любой)", value="", placeholder="Например: Москва, Санкт-Петербург")
    
    st.subheader("📌 Ключевые слова в должности")
    pos_kw_input = st.text_input(
        "Введите через запятую",
        value="Backend, Python, Developer",
        placeholder="Backend, Python, Developer, Senior"
    )
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    st.subheader("🛠️ Ключевые навыки")
    skill_kw_input = st.text_input(
        "Введите через запятую",
        value="Python, FastAPI, PostgreSQL, Docker",
        placeholder="Python, FastAPI, PostgreSQL, Docker, Kubernetes"
    )
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.subheader("🎯 Пороговые значения")
    it_threshold = st.number_input("IT-порог (0-10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    min_score = st.number_input("Минимальный semantic score (10-50)", min_value=10.0, max_value=50.0, value=15.0, step=1.0)
    top_k = st.slider("Топ кандидатов для вывода", min_value=5, max_value=50, value=20, step=5)

# ====================== Основная область ======================
vacancy_text = """Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой (готовы рассматривать релокацию)"""

st.info("📄 Текст вакансии предустановлен")

uploaded = st.file_uploader("📁 Загрузите файл с резюме (resumes_generated.txt)", type="txt")

if st.button("🔥 Запустить подбор", type="primary", use_container_width=True):
    if not uploaded:
        st.error("❌ Пожалуйста, загрузите файл с резюме")
        st.stop()
    
    # Сохраняем загруженный файл
    with open("resumes_generated.txt", "wb") as f:
        f.write(uploaded.getbuffer())
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
    hr_core.USE_AUTO_FILTERS = False  # Отключаем авто-фильтры
    hr_core.USE_MANUAL_FILTERS = True  # Используем только ручные
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(city_input.strip())
    hr_core.USE_POSITION_KEYWORDS = bool(pos_kw)
    hr_core.USE_SKILLS_KEYWORDS = bool(skill_kw)
    hr_core.USE_FILTERS = True
    hr_core.USE_RERANKING = True
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold
    
    # Загрузка резюме
    with st.spinner("📂 Загружаем резюме..."):
        resumes = hr_core.load_resumes_from_file(resume_file)
    
    # Захватываем вывод, но фильтруем лишнее
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    
    with st.spinner("🧠 ИИ анализирует кандидатов (15-40 секунд)..."):
        top_candidates = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=resumes,
            manual_filters=manual_filters,
            use_filters=True,
            use_reranking=True,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )
    
    sys.stdout = old_stdout
    console_log = captured.getvalue()
    
    # Фильтруем лог от служебных сообщений
    filtered_lines = []
    for line in console_log.split('\n'):
        skip = False
        for phrase in [
            "семантическое ранжирование",
            "reranking",
            "автоматическое извлечение",
            "ручные фильтры",
            "применяемые фильтры",
            "валидацию и фильтрацию",
            "semantic search"
        ]:
            if phrase.lower() in line.lower():
                skip = True
                break
        if not skip and line.strip():
            filtered_lines.append(line)
    clean_log = '\n'.join(filtered_lines)
    
    # ====================== Результаты ======================
    if top_candidates:
        st.success(f"✅ Найдено {len(top_candidates)} подходящих кандидатов")
        
        df = pd.DataFrame(top_candidates)
        display_cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary']
        display_cols = [c for c in display_cols if c in df.columns]
        
        st.subheader("🏆 ТОП кандидатов")
        
        # Форматируем для красивого отображения
        df_display = df[display_cols].copy()
        rename_map = {
            'name': 'Имя',
            'parsed_age': 'Возраст',
            'parsed_experience': 'Опыт (лет)',
            'city': 'Город',
            'score': 'Score (%)',
            'rerank_score_percent': 'Rerank (%)',
            'salary': 'Зарплата'
        }
        df_display = df_display.rename(columns=rename_map)
        
        # Округляем проценты
        for col in ['Score (%)', 'Rerank (%)']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(1)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Детальная информация по каждому кандидату
        st.subheader("📋 Детальная информация")
        for idx, candidate in enumerate(top_candidates[:5]):  # Показываем топ-5 детально
            with st.expander(f"🔹 {idx+1}. {candidate.get('name', 'Unknown')} — Score: {candidate.get('score', 0):.1f}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Возраст:** {candidate.get('parsed_age', '—')}")
                    st.write(f"**Опыт:** {candidate.get('parsed_experience', '—')} лет")
                    st.write(f"**Город:** {candidate.get('city', '—')}")
                with col2:
                    st.write(f"**Зарплата:** {candidate.get('salary', '—')}")
                    st.write(f"**Rerank:** {candidate.get('rerank_score_percent', 0):.1f}%")
                
                if 'skills' in candidate and candidate['skills']:
                    st.write(f"**Навыки:** {', '.join(candidate['skills'][:10])}")
    else:
        st.warning("⚠️ Не найдено кандидатов, соответствующих критериям")
    
    if clean_log and len(clean_log) > 0:
        with st.expander("📋 Детальный лог обработки"):
            st.text(clean_log)

# Восстанавливаем оригинальный print
builtins.print = original_print

st.caption("AI HR Скринер • Семантический поиск кандидатов")
