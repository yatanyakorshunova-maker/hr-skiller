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
        min_age = st.number_input("От (лет)", min_value=14, max_value=100, value=23, step=1, key="min_age")
    with col2:
        max_age = st.number_input("До (лет)", min_value=14, max_value=100, value=45, step=1, key="max_age")
    
    st.subheader("💼 Опыт работы")
    min_exp = st.number_input("Минимальный опыт (лет)", min_value=0, max_value=80, value=5, step=1, key="min_exp")
    
    st.subheader("📍 Город")
    city_input = st.text_input("Город (оставьте пустым если любой)", value="", key="city_input")
    
    st.subheader("📌 Ключевые слова в должности")
    pos_kw_input = st.text_input(
        "Введите через запятую",
        value="",
        placeholder="Backend, Python, Developer",
        key="pos_kw_input"
    )
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    st.subheader("🛠️ Ключевые навыки")
    skill_kw_input = st.text_input(
        "Введите через запятую",
        value="",
        placeholder="Python, FastAPI, PostgreSQL, Docker",
        key="skill_kw_input"
    )
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.subheader("🎯 Пороговые значения")
    it_threshold = st.number_input("IT-порог (0-10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="it_threshold")
    min_score = st.number_input("Минимальный semantic score (10-50)", min_value=10.0, max_value=50.0, value=15.0, step=1.0, key="min_score")
    top_k = st.slider("Топ кандидатов для вывода", min_value=5, max_value=50, value=20, step=5, key="top_k")

# ====================== Основная область ======================
# Текст вакансии (скрыт от пользователя)
vacancy_text = """Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой (готовы рассматривать релокацию)"""

uploaded = st.file_uploader("📁 Загрузите файл с резюме (resumes_generated.txt)", type="txt", key="file_uploader")

if st.button("🔥 Запустить подбор", type="primary", use_container_width=True, key="run_button"):
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
    hr_core.USE_AUTO_FILTERS = False
    hr_core.USE_MANUAL_FILTERS = True
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
    
    # ====================== РЕЗУЛЬТАТЫ ======================
    if top_candidates:
        # ===== СОРТИРУЕМ КАНДИДАТОВ ПО SCORE (ОТ БОЛЬШЕГО К МЕНЬШЕМУ) =====
        top_candidates_sorted = sorted(top_candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        st.success(f"✅ Найдено {len(top_candidates_sorted)} подходящих кандидатов")
        
        # Преобразуем навыки в читаемый формат
        for candidate in top_candidates_sorted:
            if 'skills' in candidate and isinstance(candidate['skills'], list):
                candidate['skills_display'] = ', '.join(candidate['skills'][:8])
                if len(candidate['skills']) > 8:
                    candidate['skills_display'] += f" (+{len(candidate['skills'])-8})"
            elif 'skills' in candidate and isinstance(candidate['skills'], str):
                candidate['skills_display'] = candidate['skills']
            else:
                candidate['skills_display'] = '—'
        
        df = pd.DataFrame(top_candidates_sorted)
        
        # Определяем колонки для отображения
        display_cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary', 'skills_display']
        display_cols = [c for c in display_cols if c in df.columns]
        
        st.subheader("🏆 ТОП кандидатов (по убыванию Score)")
        
        # Форматируем для красивого отображения
        df_display = df[display_cols].copy()
        rename_map = {
            'name': 'Имя',
            'parsed_age': 'Возраст',
            'parsed_experience': 'Опыт (лет)',
            'city': 'Город',
            'score': 'Score (%)',
            'rerank_score_percent': 'Rerank (%)',
            'salary': 'Зарплата',
            'skills_display': 'Навыки'
        }
        df_display = df_display.rename(columns=rename_map)
        
        # Округляем проценты
        for col in ['Score (%)', 'Rerank (%)']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(1)
        
        # Применяем стили для лучшего отображения
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Навыки": st.column_config.TextColumn("Навыки", width="large"),
                "Зарплата": st.column_config.TextColumn("Зарплата", width="medium"),
            }
        )
        
        # ===== ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО ВСЕМ КАНДИДАТАМ (ОТСОРТИРОВАННАЯ) =====
        st.subheader(f"📋 Детальная информация (все {len(top_candidates_sorted)} кандидатов, отсортировано по Score)")
        
        for idx, candidate in enumerate(top_candidates_sorted):
            # Формируем заголовок с номером резюме, именем и Score
            resume_id = candidate.get('resume_id', candidate.get('id', candidate.get('file_name', '')))
            resume_id_str = f" [ID: {resume_id}]" if resume_id else ""
            
            with st.expander(f"🔹 {idx+1}. {candidate.get('name', 'Unknown')}{resume_id_str} — Score: {candidate.get('score', 0):.1f}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Возраст:** {candidate.get('parsed_age', '—')}")
                    st.write(f"**Опыт:** {candidate.get('parsed_experience', '—')} лет")
                    st.write(f"**Город:** {candidate.get('city', '—')}")
                with col2:
                    st.write(f"**Зарплата:** {candidate.get('salary', '—')}")
                    st.write(f"**Rerank:** {candidate.get('rerank_score_percent', 0):.1f}%")
                    st.write(f"**Желаемая должность:** {candidate.get('position', '—')}")
                
                # Номер резюме (если есть отдельное поле)
                if 'resume_number' in candidate and candidate['resume_number']:
                    st.write(f"**Номер резюме:** {candidate['resume_number']}")
                elif 'file_name' in candidate and candidate['file_name']:
                    st.write(f"**Файл:** {candidate['file_name']}")
                
                # Навыки
                if 'skills' in candidate:
                    if isinstance(candidate['skills'], list):
                        skills_str = ', '.join(candidate['skills'])
                    else:
                        skills_str = str(candidate['skills'])
                    st.write(f"**Навыки:** {skills_str}")
                
                # Образование
                if 'education' in candidate and candidate['education']:
                    st.write(f"**Образование:** {candidate['education'][:200]}")
                
                # Последнее место работы
                if 'last_job' in candidate and candidate['last_job']:
                    st.write(f"**Последнее место работы:** {candidate['last_job']}")
                
                # Комментарий
                if 'comment' in candidate and candidate['comment']:
                    st.write(f"**Комментарий:** {candidate['comment']}")
    else:
        st.warning("⚠️ Не найдено кандидатов, соответствующих критериям")
    
    if clean_log and len(clean_log) > 0:
        with st.expander("📋 Детальный лог обработки"):
            st.text(clean_log)

# Восстанавливаем оригинальный print
builtins.print = original_print

st.caption("AI HR Скринер • Семантический поиск кандидатов")
