import streamlit as st
import pandas as pd
import io
import sys
from typing import List, Dict

import hr_core
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

# глушим вывод
import builtins
original_print = builtins.print

def silenced_print(*args, **kwargs):
    msg = str(args[0]) if args else ""
    bad_words = [
        "Выполняется семантическое ранжирование",
        "Запускаем reranking",
        "Автоматическое извлечение фильтров",
        "Применяются ручные фильтры",
        "Применяемые фильтры",
        "Начинаем валидацию и фильтрацию",
        "semantic search",
        "reranking"
    ]
    if any(w.lower() in msg.lower() for w in bad_words):
        return
    original_print(*args, **kwargs)

builtins.print = silenced_print

# настройки
st.set_page_config(page_title="AI HR Скринер", layout="wide")
st.title("🚀 AI HR Скринер")

# ===== САЙДБАР =====
with st.sidebar:
    st.header("⚙️ Настройки фильтрации")
    
    st.subheader("👤 Возраст")
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.number_input("От", min_value=14, max_value=100, value=23, step=1)
    with col2:
        max_age = st.number_input("До", min_value=14, max_value=100, value=45, step=1)
    
    st.subheader("💼 Опыт")
    min_exp = st.number_input("Мин. опыт (лет)", min_value=0, max_value=80, value=5, step=1)
    
    st.subheader("📍 Город")
    city_input = st.text_input("Город (пусто - любой)", value="")
    
    st.subheader("📌 Ключевые слова в должности")
    pos_kw_input = st.text_input("Через запятую", value="", placeholder="Backend, Python")
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    st.subheader("🛠️ Навыки")
    skill_kw_input = st.text_input("Через запятую", value="", placeholder="Python, FastAPI, Docker")
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.subheader("🎯 Пороги")
    it_threshold = st.number_input("IT-порог (0-10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    min_score = st.number_input("Min score (10-50)", min_value=10.0, max_value=50.0, value=15.0, step=1.0)
    top_k = st.slider("Топ кандидатов", min_value=5, max_value=50, value=20, step=5)

# ===== ОСНОВНАЯ ОБЛАСТЬ =====

# пустой блок для ввода вакансии
st.subheader("📝 Текст вакансии")
vacancy_text = st.text_area(
    "Введите описание вакансии",
    value="",
    height=200,
    placeholder="Например:\nИщем Python разработчика...\nТребования:\n- Опыт от 3 лет\n- Знание Django\n- PostgreSQL",
    key="vacancy_input"
)

st.divider()

# загрузка файла
uploaded = st.file_uploader("📁 Загрузите файл с резюме (resumes_generated.txt)", type="txt")

# кнопка
if st.button("🔥 Запустить подбор", type="primary", use_container_width=True):
    if not uploaded:
        st.error("❌ Нет файла с резюме")
        st.stop()
    
    if not vacancy_text.strip():
        st.error("❌ Введите текст вакансии")
        st.stop()
    
    # сохраняем файл
    with open("resumes_generated.txt", "wb") as f:
        f.write(uploaded.getbuffer())
    
    # фильтры
    manual_filters = {
        'min_age': min_age,
        'max_age': max_age,
        'min_experience': min_exp,
        'city': city_input.strip() if city_input.strip() else None,
        'position_keywords': pos_kw,
        'skills_keywords': skill_kw
    }
    
    # настройки hr_core
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
    
    # грузим резюме
    with st.spinner("📂 Загружаем..."):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
    
    # ловим вывод
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    
    with st.spinner("🧠 Анализ (15-40 сек)..."):
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
    
    # чистим лог
    clean_lines = []
    for line in console_log.split('\n'):
        skip = False
        for phrase in ["семантическое ранжирование", "reranking", "автоматическое извлечение", "ручные фильтры", "применяемые фильтры", "валидацию и фильтрацию", "semantic search"]:
            if phrase.lower() in line.lower():
                skip = True
                break
        if not skip and line.strip():
            clean_lines.append(line)
    clean_log = '\n'.join(clean_lines)
    
    # результаты
    if top_candidates:
        top_candidates_sorted = sorted(top_candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        st.success(f"✅ Найдено {len(top_candidates_sorted)} кандидатов")
        
        # чистим навыки
        for c in top_candidates_sorted:
            if 'skills' in c and isinstance(c['skills'], list):
                c['skills_display'] = ', '.join(c['skills'][:8])
                if len(c['skills']) > 8:
                    c['skills_display'] += f" (+{len(c['skills'])-8})"
            elif 'skills' in c and isinstance(c['skills'], str):
                c['skills_display'] = c['skills']
            else:
                c['skills_display'] = '—'
        
        df = pd.DataFrame(top_candidates_sorted)
        
        cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary', 'skills_display']
        cols = [c for c in cols if c in df.columns]
        
        st.subheader("🏆 ТОП кандидатов")
        
        df_display = df[cols].copy()
        df_display = df_display.rename(columns={
            'name': 'Имя', 'parsed_age': 'Возраст', 'parsed_experience': 'Опыт', 'city': 'Город',
            'score': 'Score', 'rerank_score_percent': 'Rerank', 'salary': 'Зарплата', 'skills_display': 'Навыки'
        })
        
        for col in ['Score', 'Rerank']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(1)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # детали
        st.subheader("📋 Детально")
        
        for idx, c in enumerate(top_candidates_sorted):
            rid = c.get('resume_id') or c.get('id') or c.get('file_name') or ''
            rid_str = f" [ID: {rid}]" if rid else ""
            
            with st.expander(f"🔹 {idx+1}. {c.get('name', 'Unknown')}{rid_str} — Score: {c.get('score', 0):.1f}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Возраст:** {c.get('parsed_age', '—')}")
                    st.write(f"**Опыт:** {c.get('parsed_experience', '—')} лет")
                    st.write(f"**Город:** {c.get('city', '—')}")
                with col2:
                    st.write(f"**Зарплата:** {c.get('salary', '—')}")
                    st.write(f"**Rerank:** {c.get('rerank_score_percent', 0):.1f}%")
                    st.write(f"**Должность:** {c.get('position', '—')}")
                
                if c.get('resume_number'):
                    st.write(f"**Номер:** {c['resume_number']}")
                elif c.get('file_name'):
                    st.write(f"**Файл:** {c['file_name']}")
                
                if 'skills' in c:
                    if isinstance(c['skills'], list):
                        st.write(f"**Навыки:** {', '.join(c['skills'])}")
                    else:
                        st.write(f"**Навыки:** {c['skills']}")
                
                if c.get('education'):
                    st.write(f"**Образование:** {c['education'][:200]}")
                
                if c.get('last_job'):
                    st.write(f"**Последнее место:** {c['last_job']}")
                
                if c.get('comment'):
                    st.write(f"**Комментарий:** {c['comment']}")
    else:
        st.warning("⚠️ Кандидатов не найдено")
    
    if clean_log:
        with st.expander("📋 Лог"):
            st.text(clean_log)

# возвращаем печать
builtins.print = original_print

st.caption("AI HR Скринер")  
