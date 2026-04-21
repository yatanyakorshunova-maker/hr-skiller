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

if 'auto_filters' not in st.session_state:
    st.session_state.auto_filters = {}

if 'use_auto_filters' not in st.session_state:
    st.session_state.use_auto_filters = True

if 'use_manual_filters' not in st.session_state:
    st.session_state.use_manual_filters = True

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
    
    use_auto = st.checkbox("🤖 Использовать авто-фильтры из вакансии", 
                           value=st.session_state.use_auto_filters,
                           key="use_auto_checkbox")
    use_manual = st.checkbox("✋ Использовать ручные фильтры", 
                             value=st.session_state.use_manual_filters,
                             key="use_manual_checkbox")
    
    st.session_state.use_auto_filters = use_auto
    st.session_state.use_manual_filters = use_manual
    
    st.divider()
    
    st.subheader("📋 Требования")
    
    # Слайдеры с привязкой к session_state
    min_age = st.slider("Минимальный возраст", 18, 60, 
                        value=st.session_state.min_age, 
                        key="min_age_slider")
    max_age = st.slider("Максимальный возраст", 20, 70, 
                        value=st.session_state.max_age, 
                        key="max_age_slider")
    min_exp = st.slider("Минимальный опыт (лет)", 0, 15, 
                        value=st.session_state.min_exp, 
                        key="min_exp_slider")
    
    city_filter = st.text_input("Город (оставь пустым = любой)", 
                                value=st.session_state.city_filter,
                                key="city_input")
    
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack", "Frontend", "DevOps"],
        default=st.session_state.position_keywords,
        key="position_multiselect"
    )
    
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "Flask", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL", "MongoDB", "Git"],
        default=st.session_state.skills_keywords,
        key="skills_multiselect"
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
        height=300,
        key="vacancy_input"
    )
    st.session_state.vacancy_text = vacancy_text

with col2:
    st.markdown("### 🎯 Действия")
    
    # Кнопка для применения фильтров из вакансии
    apply_filters_btn = st.button(
        "📥 Применить фильтры из вакансии", 
        type="secondary",
        use_container_width=True,
        help="Извлечь фильтры из текста вакансии и применить их"
    )
    
    st.markdown("---")
    
    reset_btn = st.button(
        "🔄 Сбросить фильтры", 
        type="secondary",
        use_container_width=True,
        help="Сбросить все фильтры к стандартным"
    )
    
    st.markdown("---")
    
    search_btn = st.button(
        "🚀 Запустить подбор", 
        type="primary",
        use_container_width=True
    )

# Обработка кнопки сброса
if reset_btn:
    st.session_state.min_age = DEFAULT_FILTERS['min_age']
    st.session_state.max_age = DEFAULT_FILTERS['max_age']
    st.session_state.min_exp = DEFAULT_FILTERS['min_exp']
    st.session_state.city_filter = DEFAULT_FILTERS['city']
    st.session_state.position_keywords = DEFAULT_FILTERS['position_keywords']
    st.session_state.skills_keywords = DEFAULT_FILTERS['skills_keywords']
    st.rerun()

# Обработка кнопки применения фильтров
if apply_filters_btn:
    with st.spinner("🔍 Извлекаю фильтры из вакансии..."):
        extracted = extract_filters_from_text(vacancy_text)
        
        # Обновляем session_state с извлеченными значениями
        if extracted.get('min_age'):
            st.session_state.min_age = extracted['min_age']
        if extracted.get('max_age'):
            st.session_state.max_age = extracted['max_age']
        if extracted.get('min_experience'):
            st.session_state.min_exp = extracted['min_experience']
        if extracted.get('city'):
            st.session_state.city_filter = extracted['city']
        if extracted.get('position_keywords') and len(extracted['position_keywords']) > 0:
            st.session_state.position_keywords = extracted['position_keywords']
        if extracted.get('skills_keywords') and len(extracted['skills_keywords']) > 0:
            st.session_state.skills_keywords = extracted['skills_keywords']
        
        st.session_state.auto_filters = extracted
        
        st.success("✅ Фильтры извлечены и применены!")
        
        # Показываем что извлечено
        st.info(f"""
        **Извлечено из вакансии:**
        - Возраст: {extracted.get('min_age', '?')} - {extracted.get('max_age', '?')} лет
        - Опыт: от {extracted.get('min_experience', '?')} лет
        - Город: {extracted.get('city', 'не указан')}
        - Должность: {', '.join(extracted.get('position_keywords', ['не указана']))}
        - Навыки: {', '.join(extracted.get('skills_keywords', ['не указаны']))}
        """)
        
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
    
    # Берем текущие значения фильтров из session_state
    manual_filters = {
        'min_age': st.session_state.min_age,
        'max_age': st.session_state.max_age,
        'min_experience': st.session_state.min_exp,
        'city': st.session_state.city_filter if st.session_state.city_filter else None,
        'position_keywords': st.session_state.position_keywords if st.session_state.position_keywords else [],
        'skills_keywords': st.session_state.skills_keywords if st.session_state.skills_keywords else []
    }
    
    # Авто-фильтры из вакансии
    auto_filters = {}
    if use_auto:
        auto_filters = extract_filters_from_text(vacancy_text)
    
    # Объединяем фильтры
    final_filters = {}
    if use_auto:
        final_filters.update(auto_filters)
    if use_manual:
        for key, value in manual_filters.items():
            if value:
                final_filters[key] = value
    
    # Настройки hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(st.session_state.city_filter) or bool(auto_filters.get('city'))
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
        
        stats = {
            'invalid_name': 0, 'invalid_city': 0, 'no_age': 0, 'no_experience': 0,
            'non_it': 0, 'filter_age': 0, 'filter_experience': 0, 'filter_city': 0,
            'filter_position': 0, 'filter_skills': 0, 'passed_all': 0
        }
        
        rejected_details = []
        use_filters_flag = use_auto or use_manual
        
        for resume in resumes:
            raw_name = str(resume.get('name', '')).strip()
            raw_city = str(resume.get('city', '')).strip()
            raw_position = str(resume.get('desired_position', '')).lower()
            raw_skills = str(resume.get('skills', '')).lower()
            
            age = None
            age_raw = resume.get('age')
            if age_raw:
                age_match = re.search(r'(\d+)', str(age_raw))
                if age_match:
                    age = int(age_match.group(1))
            
            experience = None
            exp_raw = resume.get('experience')
            if exp_raw:
                exp_match = re.search(r'(\d+)', str(exp_raw))
                if exp_match:
                    experience = int(exp_match.group(1))
            
            if not raw_name or len(raw_name) < 3:
                stats['invalid_name'] += 1
                rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': 'Некорректное имя'})
            elif age is None or not (16 <= age <= 70):
                stats['no_age'] += 1
                rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': f'Некорректный возраст: {resume.get("age", "?")}'})
            elif experience is None or not (0 <= experience <= 50):
                stats['no_experience'] += 1
                rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': f'Некорректный опыт: {resume.get("experience", "?")}'})
            elif not is_valid_russian_city(raw_city):
                stats['invalid_city'] += 1
                rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': f'Неизвестный город'})
            else:
                is_it, it_reason = is_it_candidate(resume, threshold=it_threshold)
                if not is_it:
                    stats['non_it'] += 1
                    rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': f'Не IT: {it_reason[:50]}'})
                else:
                    hr_reason = None
                    if use_filters_flag and final_filters:
                        f = final_filters
                        if f.get('min_age') and age < f['min_age']:
                            hr_reason = f"Возраст {age} < {f['min_age']}"
                            stats['filter_age'] += 1
                        elif f.get('max_age') and age > f['max_age']:
                            hr_reason = f"Возраст {age} > {f['max_age']}"
                            stats['filter_age'] += 1
                        elif f.get('min_experience') and experience < f['min_experience']:
                            hr_reason = f"Опыт {experience} < {f['min_experience']}"
                            stats['filter_experience'] += 1
                        elif f.get('city') and f['city'].lower() not in normalize_city(raw_city).lower():
                            hr_reason = f"Город '{raw_city}' ≠ '{f['city']}'"
                            stats['filter_city'] += 1
                        elif f.get('position_keywords') and len(f['position_keywords']) > 0:
                            if not any(kw.lower() in raw_position for kw in f['position_keywords']):
                                hr_reason = "Должность не подходит"
                                stats['filter_position'] += 1
                        elif f.get('skills_keywords') and len(f['skills_keywords']) > 0:
                            if not any(kw.lower() in raw_skills for kw in f['skills_keywords']):
                                hr_reason = "Навыки не подходят"
                                stats['filter_skills'] += 1
                    
                    if hr_reason:
                        rejected_details.append({'name': raw_name[:35], 'city': raw_city[:20], 'age': age or '?', 'experience': experience or '?', 'reason': hr_reason})
                    else:
                        stats['passed_all'] += 1
        
        top_candidates = hr_core.rank_candidates(
            vacancy_text=vacancy_text,
            resumes=resumes,
            manual_filters=manual_filters if use_manual else None,
            use_filters=use_filters_flag,
            use_reranking=use_reranking,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )
    
    # Вывод результатов
    st.success(f"✅ Готово! Найдено {len(top_candidates)} лучших кандидатов")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Всего", original_count)
    with col2:
        st.metric("✅ Прошло", stats['passed_all'])
    with col3:
        st.metric("❌ Отсеяно", original_count - stats['passed_all'])
    with col4:
        if original_count > 0:
            st.metric("📊 % отбора", f"{stats['passed_all']/original_count*100:.1f}%")
    
    with st.expander("🔍 Примененные фильтры", expanded=True):
        clean_filters = {k: v for k, v in final_filters.items() if v}
        if clean_filters:
            st.json(clean_filters)
        else:
            st.info("Фильтры не применены")
    
    with st.expander("📊 Статистика отсева", expanded=True):
        if rejected_details:
            st.dataframe(pd.DataFrame(rejected_details), use_container_width=True, hide_index=True)
    
    if top_candidates:
        st.subheader("🏆 ТОП кандидатов")
        df = pd.DataFrame(top_candidates)
        cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'salary']
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Не найдено подходящих кандидатов")

st.caption("Сделано на Streamlit + SentenceTransformer • AI HR Скринер 2026")