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

if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

# Инициализация значений фильтров в session_state
if 'min_age_val' not in st.session_state:
    st.session_state.min_age_val = 23
if 'max_age_val' not in st.session_state:
    st.session_state.max_age_val = 45
if 'min_exp_val' not in st.session_state:
    st.session_state.min_exp_val = 4
if 'city_filter_val' not in st.session_state:
    st.session_state.city_filter_val = ""
if 'position_keywords_val' not in st.session_state:
    st.session_state.position_keywords_val = ["Backend", "Python", "Developer"]
if 'skills_keywords_val' not in st.session_state:
    st.session_state.skills_keywords_val = ["Python", "FastAPI", "PostgreSQL", "Docker"]

# ====================== ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ФИЛЬТРОВ ИЗ ВАКАНСИИ ======================
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
    
    # Если не нашли диапазон, ищем минимальный возраст
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
    if not city_match:
        city_match = re.search(r'в\s+([А-Яа-яЁё]+)\s+(?:,|\.|\))', text_lower)
    
    if city_match:
        city_name = city_match.group(1).strip()
        if is_valid_russian_city(city_name):
            filters['city'] = city_name.title()
    
    # Извлечение ключевых слов должности
    position_list = ['backend', 'python', 'developer', 'middle', 'senior', 'fullstack', 'frontend', 'devops', 'data scientist', 'ml engineer']
    for pos in position_list:
        if pos in text_lower:
            # Преобразуем в формат как в мультиселекте
            if pos == 'backend':
                filters['position_keywords'].append('Backend')
            elif pos == 'python':
                filters['position_keywords'].append('Python')
            elif pos == 'developer':
                filters['position_keywords'].append('Developer')
            elif pos == 'middle':
                filters['position_keywords'].append('Middle')
            elif pos == 'senior':
                filters['position_keywords'].append('Senior')
            elif pos == 'fullstack':
                filters['position_keywords'].append('Fullstack')
            elif pos == 'frontend':
                filters['position_keywords'].append('Frontend')
            else:
                filters['position_keywords'].append(pos.title())
    
    # Извлечение навыков
    skill_list = {
        'python': 'Python', 'fastapi': 'FastAPI', 'django': 'Django', 'flask': 'Flask',
        'sql': 'SQL', 'postgresql': 'PostgreSQL', 'docker': 'Docker', 'kubernetes': 'Kubernetes',
        'redis': 'Redis', 'mongodb': 'MongoDB', 'git': 'Git', 'pytest': 'Pytest'
    }
    
    for skill_lower, skill_title in skill_list.items():
        if skill_lower in text_lower:
            filters['skills_keywords'].append(skill_title)
    
    # Убираем дубликаты
    filters['position_keywords'] = list(dict.fromkeys(filters['position_keywords']))
    filters['skills_keywords'] = list(dict.fromkeys(filters['skills_keywords']))
    
    return filters

def update_filters_in_sidebar(filters: Dict):
    """Обновляет значения фильтров в session_state"""
    if filters.get('min_age'):
        st.session_state.min_age_val = filters['min_age']
    if filters.get('max_age'):
        st.session_state.max_age_val = filters['max_age']
    if filters.get('min_experience'):
        st.session_state.min_exp_val = filters['min_experience']
    if filters.get('city'):
        st.session_state.city_filter_val = filters['city']
    if filters.get('position_keywords') and len(filters['position_keywords']) > 0:
        st.session_state.position_keywords_val = filters['position_keywords']
    if filters.get('skills_keywords') and len(filters['skills_keywords']) > 0:
        st.session_state.skills_keywords_val = filters['skills_keywords']

# ====================== Сайдбар с фильтрами ======================
with st.sidebar:
    st.header("⚙️ Настройки фильтров")
    
    use_auto = st.checkbox("🤖 Использовать авто-фильтры из вакансии", value=True, 
                           help="Автоматически извлекает возраст, опыт, город, ключевые слова из текста вакансии")
    use_manual = st.checkbox("✋ Использовать ручные фильтры", value=True)
    
    st.divider()
    
    st.subheader("📋 Ручные требования")
    
    # Возраст с синхронизацией
    min_age = st.number_input("Минимальный возраст", 18, 60, value=st.session_state.min_age_val, key="min_age_input")
    max_age = st.number_input("Максимальный возраст", 20, 70, value=st.session_state.max_age_val, key="max_age_input")
    
    # Опыт с синхронизацией
    min_exp = st.number_input("Минимальный опыт (лет)", 0, 15, value=st.session_state.min_exp_val, key="min_exp_input")
    
    # Город с синхронизацией
    city_filter = st.text_input("Город (оставь пустым = любой)", value=st.session_state.city_filter_val, key="city_filter_input")
    
    # Должность с синхронизацией
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack", "Frontend", "Engineer"],
        default=st.session_state.position_keywords_val,
        key="position_keywords_input"
    )
    
    # Навыки с синхронизацией
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "Flask", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL", "MongoDB", "Git", "Pytest"],
        default=st.session_state.skills_keywords_val,
        key="skills_keywords_input"
    )
    
    st.divider()
    
    it_threshold = st.slider("IT Threshold (чувствительность)", 0.0, 10.0, 2.0, 0.5)
    use_reranking = st.checkbox("Включить reranking (более точный)", value=True)
    min_score = st.slider("Минимальный semantic score (%)", 10, 40, 15)
    top_k = st.slider("Сколько кандидатов показывать", 5, 50, 20)

# Обновляем session_state из сайдбара (для синхронизации)
st.session_state.min_age_val = min_age
st.session_state.max_age_val = max_age
st.session_state.min_exp_val = min_exp
st.session_state.city_filter_val = city_filter
st.session_state.position_keywords_val = position_keywords
st.session_state.skills_keywords_val = skills_keywords

# ====================== ОСНОВНАЯ ЧАСТЬ ======================

# Создаем две колонки для вакансии
col1, col2 = st.columns([3, 1])

with col1:
    vacancy_text = st.text_area(
        "📝 Вставь текст вакансии сюда",
        value=st.session_state.vacancy_text,
        height=300,
        key="vacancy_input"
    )

with col2:
    st.markdown("### 🎯 Действия")
    
    # Кнопка для анализа вакансии
    analyze_vacancy_btn = st.button(
        "🔍 Проанализировать вакансию", 
        type="secondary",
        use_container_width=True,
        help="Нажми, чтобы автоматически извлечь требования из текста вакансии и заполнить фильтры"
    )
    
    st.markdown("---")
    
    # Кнопка для сброса фильтров
    reset_filters_btn = st.button(
        "🔄 Сбросить фильтры к стандартным", 
        type="secondary",
        use_container_width=True,
        help="Сбросить все фильтры к значениям по умолчанию"
    )
    
    st.markdown("---")
    
    # Кнопка для запуска поиска
    search_btn = st.button(
        "🚀 Запустить подбор кандидатов", 
        type="primary",
        use_container_width=True
    )

# Обработка кнопки "Сбросить фильтры"
if reset_filters_btn:
    st.session_state.min_age_val = 23
    st.session_state.max_age_val = 45
    st.session_state.min_exp_val = 4
    st.session_state.city_filter_val = ""
    st.session_state.position_keywords_val = ["Backend", "Python", "Developer"]
    st.session_state.skills_keywords_val = ["Python", "FastAPI", "PostgreSQL", "Docker"]
    st.rerun()

# Обработка кнопки "Проанализировать вакансию"
if analyze_vacancy_btn:
    with st.spinner("🔍 Анализирую вакансию и извлекаю фильтры..."):
        st.session_state.auto_filters = extract_filters_from_text(vacancy_text)
        st.session_state.vacancy_text = vacancy_text
        st.session_state.filters_applied = True
        
        # Обновляем фильтры в сайдбаре
        update_filters_in_sidebar(st.session_state.auto_filters)
        
        st.success("✅ Фильтры извлечены из вакансии и применены к панели фильтров!")
        
        # Показываем извлеченные фильтры
        st.subheader("📊 Извлеченные из вакансии фильтры:")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.auto_filters.get('min_age') or st.session_state.auto_filters.get('max_age'):
                age_str = f"{st.session_state.auto_filters.get('min_age', '?')} - {st.session_state.auto_filters.get('max_age', '?')}"
                st.info(f"📅 Возраст: {age_str} лет")
            else:
                st.info("📅 Возраст: не указан")
            
            if st.session_state.auto_filters.get('min_experience'):
                st.info(f"💼 Опыт: от {st.session_state.auto_filters['min_experience']} лет")
            else:
                st.info("💼 Опыт: не указан")
            
            if st.session_state.auto_filters.get('city'):
                st.info(f"📍 Город: {st.session_state.auto_filters['city']}")
            else:
                st.info("📍 Город: не указан")
        
        with col_b:
            if st.session_state.auto_filters.get('position_keywords'):
                st.info(f"💼 Должность: {', '.join(st.session_state.auto_filters['position_keywords'][:5])}")
            else:
                st.info("💼 Должность: не указана")
            
            if st.session_state.auto_filters.get('skills_keywords'):
                skills_preview = ', '.join(st.session_state.auto_filters['skills_keywords'][:5])
                if len(st.session_state.auto_filters['skills_keywords']) > 5:
                    skills_preview += f" +{len(st.session_state.auto_filters['skills_keywords']) - 5}"
                st.info(f"🛠️ Навыки: {skills_preview}")
            else:
                st.info("🛠️ Навыки: не указаны")
        
        st.rerun()

# Загрузка файла с резюме
uploaded_file = st.file_uploader("📁 Загрузи файл с резюме (resumes_generated.txt)", type=["txt"])

# Обработка кнопки "Запустить подбор"
if search_btn:
    if uploaded_file is None:
        st.error("❌ Сначала загрузи файл с резюме!")
        st.stop()
    
    # Сохраняем загруженный файл
    with open("resumes_generated.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Файл с резюме загружен!")
    
    # Формируем фильтры (берем текущие значения из сайдбара)
    manual_filters = {
        'min_age': min_age,
        'max_age': max_age,
        'min_experience': min_exp,
        'city': city_filter.strip() if city_filter.strip() else None,
        'position_keywords': position_keywords if position_keywords else [],
        'skills_keywords': skills_keywords if skills_keywords else []
    }
    
    # Если включены авто-фильтры, извлекаем их из вакансии
    auto_filters = {}
    if use_auto:
        auto_filters = extract_filters_from_text(vacancy_text)
        st.info(f"🤖 Авто-фильтры из вакансии: возраст {auto_filters.get('min_age', '?')}-{auto_filters.get('max_age', '?')}, опыт от {auto_filters.get('min_experience', '?')} лет")
    
    # Объединяем фильтры (авто + ручные)
    final_filters = {}
    if use_auto:
        final_filters.update(auto_filters)
    if use_manual:
        for key, value in manual_filters.items():
            if value:
                final_filters[key] = value
    
    # Применяем настройки к hr_core
    hr_core.USE_AUTO_FILTERS = use_auto
    hr_core.USE_MANUAL_FILTERS = use_manual
    hr_core.USE_MIN_AGE = True
    hr_core.USE_MAX_AGE = True
    hr_core.USE_MIN_EXPERIENCE = True
    hr_core.USE_CITY = bool(city_filter.strip()) or bool(auto_filters.get('city'))
    hr_core.USE_POSITION_KEYWORDS = True
    hr_core.USE_SKILLS_KEYWORDS = True
    hr_core.USE_FILTERS = use_auto or use_manual
    hr_core.USE_RERANKING = use_reranking
    hr_core.MIN_SCORE = min_score
    hr_core.TOP_K = top_k
    hr_core.IT_THRESHOLD = it_threshold
    
    # Запуск анализа
    with st.spinner("📊 Загружаем резюме и анализируем кандидатов... (может занять 20–60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        original_count = len(resumes)
        
        # Собираем статистику
        stats = {
            'invalid_name': 0, 'invalid_city': 0, 'no_age': 0, 'no_experience': 0,
            'non_it': 0, 'filter_age': 0, 'filter_experience': 0, 'filter_city': 0,
            'filter_position': 0, 'filter_skills': 0, 'passed_all': 0
        }
        
        rejected_details = []
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
            
            # Валидация
            if not raw_name or len(raw_name) < 3:
                reason = f"Некорректное имя: '{raw_name[:30]}'"
                category = "invalid_name"
                stats['invalid_name'] += 1
            elif age is None or not (16 <= age <= 70):
                reason = f"Некорректный возраст: {resume.get('age', 'не указан')}"
                category = "no_age"
                stats['no_age'] += 1
            elif experience is None or not (0 <= experience <= 50):
                reason = f"Некорректный опыт: {resume.get('experience', 'не указан')}"
                category = "no_experience"
                stats['no_experience'] += 1
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
                    # HR-фильтры
                    hr_filter_reason = None
                    
                    if use_filters_flag and final_filters:
                        f = final_filters
                        if f.get('min_age') and age < f['min_age']:
                            hr_filter_reason = f"Возраст {age} < {f['min_age']}"
                            category = "filter_age"
                            stats['filter_age'] += 1
                        elif f.get('max_age') and age > f['max_age']:
                            hr_filter_reason = f"Возраст {age} > {f['max_age']}"
                            category = "filter_age"
                            stats['filter_age'] += 1
                        elif f.get('min_experience') and experience < f['min_experience']:
                            hr_filter_reason = f"Опыт {experience} < {f['min_experience']}"
                            category = "filter_experience"
                            stats['filter_experience'] += 1
                        elif f.get('city') and f['city'].lower() not in normalize_city(raw_city).lower():
                            hr_filter_reason = f"Город '{raw_city}' ≠ '{f['city']}'"
                            category = "filter_city"
                            stats['filter_city'] += 1
                        elif f.get('position_keywords') and len(f['position_keywords']) > 0:
                            if not any(kw.lower() in raw_position for kw in f['position_keywords']):
                                hr_filter_reason = "Должность не содержит ключевых слов"
                                category = "filter_position"
                                stats['filter_position'] += 1
                        elif f.get('skills_keywords') and len(f['skills_keywords']) > 0:
                            if not any(kw.lower() in raw_skills for kw in f['skills_keywords']):
                                hr_filter_reason = "Навыки не содержат ключевых слов"
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
            manual_filters=manual_filters if use_manual else None,
            use_filters=use_filters_flag,
            use_reranking=use_reranking,
            min_score=min_score,
            top_k=top_k,
            it_threshold=it_threshold
        )
    
    # ====================== ВЫВОД РЕЗУЛЬТАТОВ ======================
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
            st.metric("📊 Процент отбора", f"{stats['passed_all']/original_count*100:.1f}%")
    
    # Показываем примененные фильтры
    with st.expander("🔍 Примененные фильтры", expanded=True):
        if final_filters:
            # Очищаем None значения
            clean_filters = {k: v for k, v in final_filters.items() if v}
            st.json(clean_filters)
        else:
            st.info("Фильтры не применены")
    
    # Детальная статистика
    with st.expander("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ОТСЕВА", expanded=True):
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
            
            for reason in reasons_data:
                percent = (reason['Количество'] / original_count) * 100
                st.progress(percent/100, text=f"{reason['Причина']}: {reason['Количество']} ({percent:.1f}%)")
        
        if rejected_details:
            st.subheader(f"📋 Отсеянные кандидаты ({len(rejected_details)})")
            st.dataframe(pd.DataFrame(rejected_details), use_container_width=True, hide_index=True)
    
    # Топ кандидатов
    if top_candidates:
        st.subheader("🏆 ТОП кандидатов")
        df = pd.DataFrame(top_candidates)
        cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'rerank_score_percent', 'salary']
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        
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

st.caption("Сделано на Streamlit + SentenceTransformer • AI HR Скринер 2026")