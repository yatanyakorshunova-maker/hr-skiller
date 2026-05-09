import streamlit as st
import pandas as pd
import io
import sys
from typing import Dict, List

import hr_core
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

# ==================== ГЛУШИМ ЛИШНИЙ ВЫВОД ====================
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

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(page_title="AI HR Помощник", layout="wide")
st.title("🤖 AI HR Помощник")

# ==================== ВЫБОР РЕЖИМА ====================
st.subheader("👥 Кто вы?")
mode = st.radio(
    "Выберите роль",
    options=["🧑‍💼 HR-менеджер (ищу сотрудника)", "👨‍💻 Соискатель (ищу работу)"],
    horizontal=True,
    help="HR ищет кандидатов, Соискатель ищет вакансии"
)

is_hr = "HR" in mode

# ==================== ИКОНКИ ДЛЯ РАЗНЫХ РЕЖИМОВ ====================
if is_hr:
    st.header("🎯 Подбор кандидатов под вакансию")
else:
    st.header("🎯 Подбор вакансий под резюме")

col_desc, col_icon = st.columns([4, 1])
with col_desc:
    if is_hr:
        st.markdown("""
        **Загрузите резюме кандидатов и опишите вакансию** — нейросеть найдет лучших специалистов.
        - 📄 Файл с резюме (формат как в `resumes_generated.txt`)
        - 📝 Текст вакансии с требованиями
        - ⚙️ Настройте фильтры под свои задачи
        """)
    else:
        st.markdown("""
        **Загрузите список вакансий и опишите свое резюме** — нейросеть найдет подходящие позиции.
        - 📄 Файл с вакансиями
        - 📝 Текст вашего резюме
        - ⚙️ Настройте фильтры под свои пожелания
        """)

st.divider()

# ==================== НАСТРОЙКИ ФИЛЬТРАЦИИ ====================
with st.sidebar:
    st.header("⚙️ Настройки фильтрации")
    
    # === ОТКУДА БРАТЬ ФИЛЬТРЫ ===
    st.subheader("🎯 Источник фильтров")
    
    filter_source = st.radio(
        "Откуда брать фильтры?",
        options=[
            "📝 Только из текста (авто)",
            "🖐️ Только ручные",
            "🤝 И то, и другое"
        ],
        help="Авто-фильтры извлекаются из текста вакансии/резюме"
    )
    
    use_auto = filter_source in ["📝 Только из текста (авто)", "🤝 И то, и другое"]
    use_manual = filter_source in ["🖐️ Только ручные", "🤝 И то, и другое"]
    
    st.divider()
    
    # === РУЧНЫЕ ФИЛЬТРЫ ===
    st.subheader("🖐️ Ручные фильтры")
    
    if is_hr:
        st.caption("Фильтры для поиска кандидатов")
    else:
        st.caption("Фильтры для поиска вакансий")
    
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.number_input("👤 Мин. возраст", min_value=14, max_value=100, value=None, placeholder="Не важно")
    with col2:
        max_age = st.number_input("👤 Макс. возраст", min_value=14, max_value=100, value=None, placeholder="Не важно")
    
    min_exp = st.number_input("💼 Мин. опыт (лет)", min_value=0, max_value=50, value=None, placeholder="Не важно")
    
    city_input = st.text_input("📍 Город", value="", placeholder="Оставьте пустым - не фильтровать")
    
    st.subheader("📌 Ключевые слова в должности")
    pos_kw_input = st.text_input("Через запятую", value="", placeholder="Backend, Python, Data Scientist")
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    st.subheader("🛠️ Ключевые навыки")
    skill_kw_input = st.text_input("Через запятую", value="", placeholder="Python, SQL, Docker")
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.divider()
    
    # === ПАРАМЕТРЫ РАНЖИРОВАНИЯ ===
    st.subheader("🎯 Параметры поиска")
    
    it_threshold = st.slider("🤖 IT-порог", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
    min_score = st.slider("📊 Мин. процент совпадения", min_value=0.0, max_value=100.0, value=15.0, step=5.0)
    top_k = st.slider("🏆 Топ результатов", min_value=5, max_value=50, value=20, step=5)
    
    use_reranking = st.checkbox("🔄 Использовать точное ранжирование (медленнее, но точнее)", value=True)

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================

# Поле для ввода текста (вакансии или резюме)
if is_hr:
    st.subheader("📝 Текст вакансии")
    vacancy_text = st.text_area(
        "Опишите требования к кандидату",
        value="",
        height=200,
        placeholder="""
Пример:
Ищем Python разработчика (Middle/Senior). Опыт от 3 лет.
Обязательные навыки: FastAPI, PostgreSQL, Docker.
Возраст: 25-40 лет.
Город: Москва или удаленно.
        """,
        key="vacancy_input"
    )
else:
    st.subheader("📝 Текст вашего резюме")
    vacancy_text = st.text_area(
        "Опишите ваш опыт и навыки",
        value="",
        height=200,
        placeholder="""
Пример:
Python разработчик, опыт 5 лет.
Навыки: FastAPI, Django, PostgreSQL, Docker, Kubernetes.
Ищу работу в Москве или удаленно.
Возраст: 28 лет.
        """,
        key="resume_input"
    )

st.divider()

# Загрузка файла
if is_hr:
    uploaded = st.file_uploader("📁 Загрузите файл с резюме (resumes_generated.txt)", type="txt")
else:
    uploaded = st.file_uploader("📁 Загрузите файл с вакансиями", type="txt")

# ==================== КНОПКА ЗАПУСКА ====================
if st.button("🔥 Начать поиск", type="primary", use_container_width=True):
    
    # Проверки
    if not uploaded:
        st.error("❌ Загрузите файл!")
        st.stop()
    
    if not vacancy_text.strip():
        st.error("❌ Введите текст!")
        st.stop()
    
    # Сохраняем файл
    filename = "resumes_generated.txt" if is_hr else "vacancies.txt"
    with open(filename, "wb") as f:
        f.write(uploaded.getbuffer())
    
    # Формируем фильтры
    manual_filters = {}
    
    if use_manual:
        if min_age is not None:
            manual_filters['min_age'] = min_age
        if max_age is not None:
            manual_filters['max_age'] = max_age
        if min_exp is not None:
            manual_filters['min_experience'] = min_exp
        if city_input.strip():
            manual_filters['city'] = city_input.strip()
        if pos_kw:
            manual_filters['position_keywords'] = pos_kw
        if skill_kw:
            manual_filters['skills_keywords'] = skill_kw
    
    # Настройки hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = min_age is not None
    hr_core.USE_MAX_AGE = max_age is not None
    hr_core.USE_MIN_EXPERIENCE = min_exp is not None
    hr_core.USE_CITY = bool(city_input.strip())
    hr_core.USE_POSITION_KEYWORDS = bool(pos_kw)
    hr_core.USE_SKILLS_KEYWORDS = bool(skill_kw)
    hr_core.USE_FILTERS = use_auto or use_manual
    hr_core.USE_RERANKING = use_reranking
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold
    
    # Загружаем данные
    with st.spinner("📂 Загружаем данные..."):
        data = hr_core.load_resumes_from_file(filename)
    
    # Запускаем анализ
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    
    with st.spinner("🧠 Анализ (15-40 сек)..."):
        results = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=data,
            manual_filters=manual_filters if use_manual else None,
            use_filters=hr_core.USE_FILTERS,
            use_reranking=use_reranking,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )
    
    sys.stdout = old_stdout
    console_log = captured.getvalue()
    
    # ==================== ПОКАЗ РЕЗУЛЬТАТОВ ====================
    if results:
        results_sorted = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        if is_hr:
            st.success(f"✅ Найдено {len(results_sorted)} подходящих кандидатов")
        else:
            st.success(f"✅ Найдено {len(results_sorted)} подходящих вакансий")
        
        # Подготовка данных для таблицы
        for item in results_sorted:
            if 'skills' in item and isinstance(item['skills'], list):
                item['skills_display'] = ', '.join(item['skills'][:5])
                if len(item['skills']) > 5:
                    item['skills_display'] += f" (+{len(item['skills'])-5})"
            elif 'skills' in item and isinstance(item['skills'], str):
                item['skills_display'] = item['skills'][:100]
            else:
                item['skills_display'] = '—'
        
        df = pd.DataFrame(results_sorted)
        
        # Выбираем колонки для отображения
        if is_hr:
            cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'skills_display']
            rename = {
                'name': 'Имя', 'parsed_age': 'Возраст', 'parsed_experience': 'Опыт',
                'city': 'Город', 'score': 'Совпадение %', 'rerank_score_percent': 'Точный %',
                'skills_display': 'Навыки'
            }
        else:
            cols = ['desired_position', 'company', 'salary', 'city', 'score', 'rerank_score_percent', 'skills_display']
            rename = {
                'desired_position': 'Должность', 'company': 'Компания', 'salary': 'Зарплата',
                'city': 'Город', 'score': 'Совпадение %', 'rerank_score_percent': 'Точный %',
                'skills_display': 'Требования'
            }
        
        # Фильтруем существующие колонки
        cols = [c for c in cols if c in df.columns]
        df_display = df[cols].copy()
        df_display = df_display.rename(columns=rename)
        
        for col in ['Совпадение %', 'Точный %']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(1)
        
        st.subheader("🏆 Результаты")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Детальный просмотр
        st.subheader("📋 Детальная информация")
        
        for idx, item in enumerate(results_sorted):
            with st.expander(f"🔹 {idx+1}. {item.get('name' if is_hr else 'desired_position', 'Unknown')} — Совпадение: {item.get('score', 0):.1f}%"):
                col1, col2 = st.columns(2)
                with col1:
                    if is_hr:
                        st.write(f"**Имя:** {item.get('name', '—')}")
                        st.write(f"**Возраст:** {item.get('parsed_age', '—')}")
                        st.write(f"**Опыт:** {item.get('parsed_experience', '—')} лет")
                    else:
                        st.write(f"**Должность:** {item.get('desired_position', '—')}")
                        st.write(f"**Компания:** {item.get('company', '—')}")
                        if item.get('requirements'):
                            st.write(f"**Требования:** {item.get('requirements', '—')[:200]}")
                with col2:
                    st.write(f"**Город:** {item.get('city', '—')}")
                    st.write(f"**Зарплата:** {item.get('salary', '—')}")
                    st.write(f"**Совпадение:** {item.get('score', 0):.1f}%")
                    if item.get('rerank_score_percent'):
                        st.write(f"**Точное совпадение:** {item.get('rerank_score_percent', 0):.1f}%")
                
                if item.get('skills'):
                    skills_str = ', '.join(item['skills']) if isinstance(item['skills'], list) else item['skills']
                    st.write(f"**{'Навыки' if is_hr else 'Требования'}:** {skills_str[:300]}")
                
                if item.get('education'):
                    st.write(f"**Образование:** {item['education'][:200]}")
    else:
        st.warning("⚠️ Ничего не найдено. Попробуйте снизить порог совпадения или ослабить фильтры.")
    
    # Показываем лог
    if console_log:
        clean_lines = []
        for line in console_log.split('\n'):
            skip = False
            for phrase in ["семантическое", "reranking", "автоматическое", "ручные", "валидацию"]:
                if phrase.lower() in line.lower():
                    skip = True
                    break
            if not skip and line.strip():
                clean_lines.append(line)
        if clean_lines:
            with st.expander("📋 Журнал работы"):
                st.text('\n'.join(clean_lines))

# ==================== ФУТЕР ====================
st.divider()
st.caption("🤖 AI HR Помощник | Работает на нейросетях | Ваши данные никуда не отправляются")

# Восстанавливаем печать
builtins.print = original_print
