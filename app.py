import streamlit as st
import pandas as pd
import re
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')

# ====================== ИМПОРТ МОДУЛЕЙ ======================
import hr_core
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

st.set_page_config(page_title="AI HR Скринер", layout="wide")
st.title("🚀 AI HR Скринер — Подбор Python Backend разработчиков")
st.markdown("Умный подбор кандидатов с фильтрами и нейросетями")

# ====================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ======================
if 'vacancy_text' not in st.session_state:
    st.session_state.vacancy_text = """Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой"""

if 'extracted_filters' not in st.session_state:
    st.session_state.extracted_filters = {}

# Значения фильтров по умолчанию
DEFAULT_FILTERS = {
    'min_age': 23,
    'max_age': 45,
    'min_exp': 4,
    'city': "",
    'position_keywords': ["Backend", "Python", "Developer"],
    'skills_keywords': ["Python", "FastAPI", "PostgreSQL", "Docker"]
}

# Инициализация фильтров в session_state
if 'min_age' not in st.session_state:
    st.session_state.min_age = DEFAULT_FILTERS['min_age']
if 'max_age' not in st.session_state:
    st.session_state.max_age = DEFAULT_FILTERS['max_age']
if 'min_exp' not in st.session_state:
    st.session_state.min_exp = DEFAULT_FILTERS['min_exp']
if 'city_filter' not in st.session_state:
    st.session_state.city_filter = DEFAULT_FILTERS['city']
if 'position_keywords' not in st.session_state:
    st.session_state.position_keywords = DEFAULT_FILTERS['position_keywords']
if 'skills_keywords' not in st.session_state:
    st.session_state.skills_keywords = DEFAULT_FILTERS['skills_keywords']

# ====================== ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ФИЛЬТРОВ ======================
def extract_filters_from_text(text: str) -> Dict:
    """Извлекает фильтры из текста вакансии"""
    filters = {
        'min_age': None,
        'max_age': None,
        'min_experience': None,
        'city': None,
        'position_keywords': [],
        'skills_keywords': []
    }
    
    text_lower = text.lower()
    
    # Извлечение возраста
    age_patterns = [
        r'возраст[:\s]*от\s*(\d{1,2})\s*до\s*(\d{1,2})',
        r'возраст[:\s]*(\d{1,2})\s*[-–—]\s*(\d{1,2})',
        r'(?:от|с)\s*(\d{1,2})\s*до\s*(\d{1,2})\s*лет',
        r'(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*лет'
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            filters['min_age'] = int(match.group(1))
            filters['max_age'] = int(match.group(2))
            break
    
    if not filters['min_age']:
        min_age_match = re.search(r'(?:возраст|от|с)\s*(\d{1,2})\s*(?:год|лет)', text_lower)
        if min_age_match:
            filters['min_age'] = int(min_age_match.group(1))
    
    # Извлечение опыта
    exp_patterns = [
        r'опыт[:\s]*от\s*(\d{1,2})\s*(?:год|лет)',
        r'стаж[:\s]*от\s*(\d{1,2})\s*(?:год|лет)',
        r'требуемый опыт[:\s]*(\d{1,2})\s*(?:год|лет)',
        r'(\d{1,2})\s*[+\+]\s*(?:год|лет)\s*опыта'
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, text_lower)
        if match:
            filters['min_experience'] = int(match.group(1))
            break
    
    # Извлечение города
    city_match = re.search(r'город[:\s]+([А-Яа-яЁё\s-]+)', text_lower)
    if not city_match:
        city_match = re.search(r'локация[:\s]+([А-Яа-яЁё\s-]+)', text_lower)
    
    if city_match:
        city_name = city_match.group(1).strip()
        if is_valid_russian_city(city_name):
            filters['city'] = city_name.title()
    
    # Извлечение ключевых слов должности
    position_map = {
        'backend': 'Backend', 'python': 'Python', 'developer': 'Developer',
        'middle': 'Middle', 'senior': 'Senior', 'fullstack': 'Fullstack',
        'frontend': 'Frontend', 'devops': 'DevOps'
    }
    
    for key, value in position_map.items():
        if key in text_lower:
            filters['position_keywords'].append(value)
    
    # Извлечение навыков
    skill_map = {
        'python': 'Python', 'fastapi': 'FastAPI', 'django': 'Django',
        'flask': 'Flask', 'postgresql': 'PostgreSQL', 'docker': 'Docker',
        'kubernetes': 'Kubernetes', 'redis': 'Redis', 'sql': 'SQL'
    }
    
    for key, value in skill_map.items():
        if key in text_lower:
            filters['skills_keywords'].append(value)
    
    # Убираем дубликаты
    filters['position_keywords'] = list(dict.fromkeys(filters['position_keywords']))
    filters['skills_keywords'] = list(dict.fromkeys(filters['skills_keywords']))
    
    return filters

# ====================== Сайдбар с фильтрами ======================
with st.sidebar:
    st.header("⚙️ Настройки фильтров")
    
    use_auto = st.checkbox("🤖 Использовать авто-фильтры из вакансии", value=True)
    use_manual = st.checkbox("✋ Использовать ручные фильтры", value=True)
    
    st.divider()
    
    st.subheader("📋 Ручные требования")
    
    min_age = st.slider("Минимальный возраст", 18, 60, value=st.session_state.min_age)
    max_age = st.slider("Максимальный возраст", 18, 70, value=st.session_state.max_age)
    min_exp = st.slider("Минимальный опыт (лет)", 0, 15, value=st.session_state.min_exp)
    
    city_filter = st.text_input("Город (оставь пустым = любой)", value=st.session_state.city_filter)
    
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack", "Frontend", "DevOps"],
        default=st.session_state.position_keywords
    )
    
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "Flask", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL", "MongoDB", "Git"],
        default=st.session_state.skills_keywords
    )
    
    # Сохраняем значения в session_state
    st.session_state.min_age = min_age
    st.session_state.max_age = max_age
    st.session_state.min_exp = min_exp
    st.session_state.city_filter = city_filter
    st.session_state.position_keywords = position_keywords
    st.session_state.skills_keywords = skills_keywords
    
    st.divider()
    
    it_threshold = st.slider("IT Threshold (чувствительность)", 0.0, 10.0, 2.0, 0.5)
    use_reranking = st.checkbox("Включить reranking (более точный)", value=True)
    min_score = st.slider("Минимальный semantic score (%)", 10, 40, 15)
    top_k = st.slider("Сколько кандидатов показывать", 5, 50, 20)

# ====================== ОСНОВНАЯ ЧАСТЬ ======================

col1, col2 = st.columns([3, 1])

with col1:
    vacancy_text = st.text_area(
        "📝 Вставь текст вакансии сюда",
        value=st.session_state.vacancy_text,
        height=250
    )
    st.session_state.vacancy_text = vacancy_text

with col2:
    st.markdown("### 🎯 Действия")
    
    analyze_btn = st.button(
        "🔍 Проанализировать вакансию", 
        type="secondary",
        use_container_width=True,
        help="Извлечь фильтры из текста вакансии"
    )
    
    reset_btn = st.button(
        "🔄 Сбросить фильтры", 
        type="secondary",
        use_container_width=True,
        help="Сбросить все фильтры к стандартным"
    )
    
    st.markdown("---")
    
    search_btn = st.button(
        "🚀 Запустить подбор кандидатов", 
        type="primary",
        use_container_width=True
    )

# ====================== ВИДЖЕТ ПРИМЕНЕННЫХ ФИЛЬТРОВ ======================
# Показываем какие фильтры сейчас активны
st.subheader("🔍 Активные фильтры")

# Определяем какие фильтры будут применены
active_filters_display = {}

if use_auto:
    auto_filters = extract_filters_from_text(vacancy_text)
    if auto_filters.get('min_age') or auto_filters.get('max_age'):
        age_str = f"{auto_filters.get('min_age', '?')} - {auto_filters.get('max_age', '?')}"
        active_filters_display["📅 Возраст (из вакансии)"] = f"{age_str} лет"
    if auto_filters.get('min_experience'):
        active_filters_display["💼 Опыт (из вакансии)"] = f"от {auto_filters['min_experience']} лет"
    if auto_filters.get('city'):
        active_filters_display["📍 Город (из вакансии)"] = auto_filters['city']
    if auto_filters.get('position_keywords'):
        active_filters_display["💼 Должность (из вакансии)"] = ", ".join(auto_filters['position_keywords'][:3])
    if auto_filters.get('skills_keywords'):
        active_filters_display["🛠️ Навыки (из вакансии)"] = ", ".join(auto_filters['skills_keywords'][:3])

if use_manual:
    if min_age:
        active_filters_display["📅 Минимальный возраст (ручной)"] = f"{min_age} лет"
    if max_age:
        active_filters_display["📅 Максимальный возраст (ручной)"] = f"{max_age} лет"
    if min_exp:
        active_filters_display["💼 Минимальный опыт (ручной)"] = f"{min_exp} лет"
    if city_filter:
        active_filters_display["📍 Город (ручной)"] = city_filter
    if position_keywords:
        active_filters_display["💼 Должность (ручная)"] = ", ".join(position_keywords[:3])
    if skills_keywords:
        active_filters_display["🛠️ Навыки (ручные)"] = ", ".join(skills_keywords[:3])

if active_filters_display:
    for key, value in active_filters_display.items():
        st.info(f"**{key}:** {value}")
else:
    st.info("ℹ️ Фильтры не применены. Включите авто-фильтры или ручные фильтры.")

st.divider()

# Обработка кнопки анализа
if analyze_btn:
    with st.spinner("🔍 Анализирую вакансию..."):
        extracted = extract_filters_from_text(vacancy_text)
        st.session_state.extracted_filters = extracted
        
        # Показываем извлеченные фильтры
        st.subheader("📊 Извлеченные из вакансии фильтры:")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if extracted.get('min_age') or extracted.get('max_age'):
                age_str = f"{extracted.get('min_age', '?')} - {extracted.get('max_age', '?')}"
                st.success(f"📅 Возраст: {age_str} лет")
            else:
                st.info("📅 Возраст: не указан")
            
            if extracted.get('min_experience'):
                st.success(f"💼 Опыт: от {extracted['min_experience']} лет")
            else:
                st.info("💼 Опыт: не указан")
            
            if extracted.get('city'):
                st.success(f"📍 Город: {extracted['city']}")
            else:
                st.info("📍 Город: не указан")
        
        with col_b:
            if extracted.get('position_keywords'):
                st.success(f"💼 Должность: {', '.join(extracted['position_keywords'][:5])}")
            else:
                st.info("💼 Должность: не указана")
            
            if extracted.get('skills_keywords'):
                skills_str = ', '.join(extracted['skills_keywords'][:5])
                st.success(f"🛠️ Навыки: {skills_str}")
            else:
                st.info("🛠️ Навыки: не указаны")
        
        st.info("💡 Нажмите 'Запустить подбор кандидатов' чтобы применить эти фильтры")

# Обработка кнопки сброса
if reset_btn:
    st.session_state.min_age = DEFAULT_FILTERS['min_age']
    st.session_state.max_age = DEFAULT_FILTERS['max_age']
    st.session_state.min_exp = DEFAULT_FILTERS['min_exp']
    st.session_state.city_filter = DEFAULT_FILTERS['city']
    st.session_state.position_keywords = DEFAULT_FILTERS['position_keywords']
    st.session_state.skills_keywords = DEFAULT_FILTERS['skills_keywords']
    st.rerun()

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузи файл с резюме (resumes_generated.txt)", type=["txt"])

# Обработка поиска
if search_btn:
    if uploaded_file is None:
        st.error("❌ Сначала загрузи файл с резюме!")
        st.stop()
    
    with open("resumes_generated.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Файл загружен!")
    
    # Формируем финальные фильтры
    final_filters = {}
    
    if use_auto:
        auto_filters = extract_filters_from_text(vacancy_text)
        # Добавляем только непустые значения
        if auto_filters.get('min_age'):
            final_filters['min_age'] = auto_filters['min_age']
        if auto_filters.get('max_age'):
            final_filters['max_age'] = auto_filters['max_age']
        if auto_filters.get('min_experience'):
            final_filters['min_experience'] = auto_filters['min_experience']
        if auto_filters.get('city'):
            final_filters['city'] = auto_filters['city']
        if auto_filters.get('position_keywords'):
            final_filters['position_keywords'] = auto_filters['position_keywords']
        if auto_filters.get('skills_keywords'):
            final_filters['skills_keywords'] = auto_filters['skills_keywords']
    
    if use_manual:
        if min_age:
            final_filters['min_age'] = min_age
        if max_age:
            final_filters['max_age'] = max_age
        if min_exp:
            final_filters['min_experience'] = min_exp
        if city_filter:
            final_filters['city'] = city_filter
        if position_keywords:
            final_filters['position_keywords'] = position_keywords
        if skills_keywords:
            final_filters['skills_keywords'] = skills_keywords
    
    # Настройки hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(city_filter) or bool(extract_filters_from_text(vacancy_text).get('city'))
    hr_core.USE_POSITION_KEYWORDS = True
    hr_core.USE_SKILLS_KEYWORDS = True
    hr_core.USE_FILTERS = use_auto or use_manual
    hr_core.USE_RERANKING = use_reranking
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold
    
    with st.spinner("📊 Анализируем кандидатов... (20-60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        original_count = len(resumes)
        
        # Фильтруем кандидатов вручную для точной статистики
        filtered_candidates = []
        rejected_candidates = []
        rejection_reasons = []
        
        for resume in resumes:
            # Извлекаем возраст
            age = None
            age_raw = resume.get('age')
            if age_raw:
                age_match = re.search(r'(\d+)', str(age_raw))
                if age_match:
                    age = int(age_match.group(1))
            
            # Извлекаем опыт
            experience = None
            exp_raw = resume.get('experience')
            if exp_raw:
                exp_match = re.search(r'(\d+)', str(exp_raw))
                if exp_match:
                    experience = int(exp_match.group(1))
            
            # Проверяем фильтры
            passed = True
            reason = None
            
            # Возраст
            if final_filters.get('min_age') and age and age < final_filters['min_age']:
                passed = False
                reason = f"Возраст {age} < {final_filters['min_age']}"
            elif final_filters.get('max_age') and age and age > final_filters['max_age']:
                passed = False
                reason = f"Возраст {age} > {final_filters['max_age']}"
            # Опыт
            elif final_filters.get('min_experience') and experience and experience < final_filters['min_experience']:
                passed = False
                reason = f"Опыт {experience} < {final_filters['min_experience']}"
            # Город
            elif final_filters.get('city') and final_filters['city'].lower() not in normalize_city(resume.get('city', '')).lower():
                passed = False
                reason = f"Город '{resume.get('city')}' ≠ '{final_filters['city']}'"
            # Должность
            elif final_filters.get('position_keywords') and len(final_filters['position_keywords']) > 0:
                pos_text = resume.get('desired_position', '').lower()
                if not any(kw.lower() in pos_text for kw in final_filters['position_keywords']):
                    passed = False
                    reason = f"Должность не содержит ключевых слов"
            # Навыки
            elif final_filters.get('skills_keywords') and len(final_filters['skills_keywords']) > 0:
                skills_text = resume.get('skills', '').lower()
                if not any(kw.lower() in skills_text for kw in final_filters['skills_keywords']):
                    passed = False
                    reason = f"Навыки не содержат ключевых слов"
            
            if passed:
                filtered_candidates.append(resume)
            else:
                rejected_candidates.append(resume)
                rejection_reasons.append({
                    'name': resume.get('name', '?')[:30],
                    'age': age,
                    'experience': experience,
                    'city': resume.get('city', '?'),
                    'reason': reason
                })
        
        # Запускаем ранжирование только для прошедших фильтр
        if filtered_candidates:
            top_candidates = hr_core.rank_candidates(
                vacancy_text=vacancy_text,
                resumes=filtered_candidates,
                manual_filters=None,  # Фильтры уже применены
                use_filters=False,  # Отключаем фильтры в rank_candidates
                use_reranking=use_reranking,
                min_score=min_score,
                top_k=top_k,
                it_threshold=it_threshold
            )
        else:
            top_candidates = []
    
    # ====================== ВЫВОД РЕЗУЛЬТАТОВ ======================
    st.success(f"✅ Готово! Найдено {len(top_candidates)} кандидатов из {original_count}")
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Всего резюме", original_count)
    with col2:
        st.metric("✅ Прошло фильтры", len(filtered_candidates))
    with col3:
        st.metric("❌ Отсеяно", len(rejected_candidates))
    with col4:
        if original_count > 0:
            st.metric("📊 Процент отбора", f"{len(filtered_candidates)/original_count*100:.1f}%")
    
    # Показываем примененные фильтры
    with st.expander("🔍 Примененные фильтры", expanded=True):
        if final_filters:
            clean_filters = {k: v for k, v in final_filters.items() if v}
            st.json(clean_filters)
        else:
            st.info("Фильтры не применены")
    
    # Показываем отсеянных
    with st.expander("📊 Отсеянные кандидаты", expanded=True):
        if rejection_reasons:
            df_rejected = pd.DataFrame(rejection_reasons)
            st.dataframe(df_rejected, use_container_width=True, hide_index=True)
        else:
            st.info("Нет отсеянных кандидатов")
    
    # Топ кандидатов
    if top_candidates:
        st.subheader("🏆 ТОП кандидатов")
        df = pd.DataFrame(top_candidates)
        cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'salary']
        cols = [c for c in cols if c in df.columns]
        
        # Добавляем цветовую индикацию
        styled_df = df[cols].copy()
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Детальная информация
        with st.expander("📋 Детальная информация о кандидатах"):
            for i, cand in enumerate(top_candidates, 1):
                st.markdown(f"**{i}. {cand.get('name', 'Неизвестно')}**")
                st.text(f"   📅 Возраст: {cand.get('parsed_age', cand.get('age', '?'))} лет")
                st.text(f"   💼 Опыт: {cand.get('parsed_experience', cand.get('experience', '?'))} лет")
                st.text(f"   📍 Город: {cand.get('city', '?')}")
                st.text(f"   💰 Зарплата: {cand.get('salary', '?')}")
                st.text(f"   🎯 Совместимость: {cand.get('score', 0)}%")
                if cand.get('rerank_score_percent'):
                    st.text(f"   🎯 Точность: {cand.get('rerank_score_percent')}%")
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

st.caption("Сделано на Streamlit + SentenceTransformer • AI HR Скринер 2026")