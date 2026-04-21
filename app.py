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

# ====================== ФУНКЦИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ЧИСЛА ИЗ СТРОКИ ======================
def extract_number_from_string(text: str) -> int:
    """Извлекает первое число из строки"""
    if not text:
        return None
    numbers = re.findall(r'(\d+)', str(text))
    if numbers:
        return int(numbers[0])
    return None

# ====================== Сайдбар с фильтрами ======================
with st.sidebar:
    st.header("⚙️ Настройки фильтров")
    
    use_auto = st.checkbox("🤖 Использовать авто-фильтры из вакансии", value=True, 
                           help="Автоматически извлекает возраст, опыт, город, ключевые слова из текста вакансии")
    use_manual = st.checkbox("✋ Использовать ручные фильтры", value=True)
    
    st.divider()
    
    st.subheader("📋 Ручные требования")
    min_age = st.number_input("Минимальный возраст", 18, 60, 23, key="min_age_input")
    max_age = st.number_input("Максимальный возраст", 20, 70, 45, key="max_age_input")
    min_exp = st.number_input("Минимальный опыт (лет)", 0, 15, 4, key="min_exp_input")
    
    city_filter = st.text_input("Город (оставь пустым = любой)", "", key="city_input")
    
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack", "Engineer"],
        default=["Backend", "Python", "Developer"],
        key="position_input"
    )
    
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "Flask", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL", "MongoDB"],
        default=["Python", "FastAPI", "PostgreSQL", "Docker"],
        key="skills_input"
    )
    
    st.divider()
    
    it_threshold = st.slider("IT Threshold (чувствительность)", 0.0, 10.0, 2.0, 0.5)
    use_reranking = st.checkbox("Включить reranking (более точный)", value=True)
    min_score = st.slider("Минимальный semantic score (%)", 10, 40, 15)
    top_k = st.slider("Сколько кандидатов показывать", 5, 50, 20)

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
    position_list = ['backend', 'python', 'developer', 'middle', 'senior', 'fullstack', 'frontend', 'devops']
    for pos in position_list:
        if pos in text_lower:
            filters['position_keywords'].append(pos.title())
    
    # Извлечение навыков
    skill_list = ['python', 'fastapi', 'django', 'flask', 'sql', 'postgresql', 'docker', 'kubernetes', 'redis']
    for skill in skill_list:
        if skill in text_lower:
            filters['skills_keywords'].append(skill.title())
    
    # Убираем дубликаты
    filters['position_keywords'] = list(dict.fromkeys(filters['position_keywords']))
    filters['skills_keywords'] = list(dict.fromkeys(filters['skills_keywords']))
    
    return filters

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
    
    analyze_vacancy_btn = st.button(
        "🔍 Проанализировать вакансию", 
        type="secondary",
        use_container_width=True,
        help="Нажми, чтобы автоматически извлечь требования из текста вакансии"
    )
    
    st.markdown("---")
    
    search_btn = st.button(
        "🚀 Запустить подбор кандидатов", 
        type="primary",
        use_container_width=True
    )

# Обработка кнопки "Проанализировать вакансию"
if analyze_vacancy_btn:
    with st.spinner("🔍 Анализирую вакансию..."):
        st.session_state.auto_filters = extract_filters_from_text(vacancy_text)
        st.session_state.vacancy_text = vacancy_text
        st.session_state.filters_applied = True
        st.success("✅ Фильтры извлечены из вакансии!")
        
        # Показываем извлеченные фильтры
        st.subheader("📊 Извлеченные фильтры:")
        
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
                st.info(f"🛠️ Навыки: {skills_preview}")
            else:
                st.info("🛠️ Навыки: не указаны")

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
    
    # Формируем финальные фильтры
    final_filters = {}
    
    # Авто-фильтры из вакансии
    if use_auto:
        auto_filters = extract_filters_from_text(vacancy_text)
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
        
        st.info(f"🤖 Авто-фильтры: возраст {auto_filters.get('min_age', '?')}-{auto_filters.get('max_age', '?')}, опыт от {auto_filters.get('min_experience', '?')} лет")
    
    # Ручные фильтры (переопределяют авто)
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
        
        st.info(f"✋ Ручные фильтры: возраст {min_age}-{max_age}, опыт от {min_exp} лет")
    
    # Показываем итоговые фильтры
    st.subheader("🔍 Итоговые фильтры:")
    if final_filters:
        st.json(final_filters)
    else:
        st.info("Фильтры не применены")
    
    # Запуск анализа
    with st.spinner("📊 Анализируем кандидатов... (20-60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        original_count = len(resumes)
        
        # ДИАГНОСТИКА: выводим загруженные резюме
        st.write("---")
        st.subheader("📋 Загруженные резюме:")
        for i, r in enumerate(resumes[:5]):
            exp_value = r.get('experience')
            exp_num = extract_number_from_string(exp_value) if exp_value else None
            st.text(f"{i+1}. {r.get('name')} | Возраст: {r.get('age')} | Опыт: {exp_value} → число: {exp_num} | Город: {r.get('city')}")
        
        # Фильтруем кандидатов
        filtered_candidates = []
        rejected_candidates = []
        
        for resume in resumes:
            # Извлекаем возраст и опыт как числа
            age_str = resume.get('age')
            exp_str = resume.get('experience')
            
            age = extract_number_from_string(age_str) if age_str else None
            experience = extract_number_from_string(exp_str) if exp_str else None
            
            # Проверяем фильтры
            passed = True
            reason = None
            
            # Проверка возраста
            if final_filters.get('min_age') and age and age < final_filters['min_age']:
                passed = False
                reason = f"Возраст {age} < {final_filters['min_age']}"
            elif final_filters.get('max_age') and age and age > final_filters['max_age']:
                passed = False
                reason = f"Возраст {age} > {final_filters['max_age']}"
            # Проверка опыта
            elif final_filters.get('min_experience') and experience and experience < final_filters['min_experience']:
                passed = False
                reason = f"Опыт {experience} < {final_filters['min_experience']}"
            # Проверка города
            elif final_filters.get('city') and final_filters['city'].lower() not in normalize_city(resume.get('city', '')).lower():
                passed = False
                reason = f"Город '{resume.get('city')}' ≠ '{final_filters['city']}'"
            # Проверка должности
            elif final_filters.get('position_keywords') and len(final_filters['position_keywords']) > 0:
                pos_text = resume.get('desired_position', '').lower()
                if not any(kw.lower() in pos_text for kw in final_filters['position_keywords']):
                    passed = False
                    reason = "Должность не содержит ключевых слов"
            # Проверка навыков
            elif final_filters.get('skills_keywords') and len(final_filters['skills_keywords']) > 0:
                skills_text = resume.get('skills', '').lower()
                if not any(kw.lower() in skills_text for kw in final_filters['skills_keywords']):
                    passed = False
                    reason = "Навыки не содержат ключевых слов"
            
            if passed:
                # Добавляем parsed поля для hr_core
                resume['parsed_age'] = age
                resume['parsed_experience'] = experience
                filtered_candidates.append(resume)
            else:
                rejected_candidates.append({
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
                manual_filters=None,
                use_filters=False,
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
    
    # Отсеянные кандидаты
    with st.expander("📊 Отсеянные кандидаты", expanded=True):
        if rejected_candidates:
            df_rejected = pd.DataFrame(rejected_candidates)
            st.dataframe(df_rejected, use_container_width=True, hide_index=True)
        else:
            st.info("Нет отсеянных кандидатов")
    
    # Топ кандидатов
    if top_candidates:
        st.subheader("🏆 ТОП кандидатов")
        df = pd.DataFrame(top_candidates)
        cols = ['name', 'parsed_age', 'parsed_experience', 'city', 'score', 'salary']
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        
        # Детальная информация
        with st.expander("📋 Детальная информация о кандидатах"):
            for i, cand in enumerate(top_candidates, 1):
                st.markdown(f"**{i}. {cand.get('name', 'Неизвестно')}**")
                st.text(f"   📅 Возраст: {cand.get('parsed_age', cand.get('age', '?'))} лет")
                st.text(f"   💼 Опыт: {cand.get('parsed_experience', cand.get('experience', '?'))} лет")
                st.text(f"   📍 Город: {cand.get('city', '?')}")
                st.text(f"   💰 Зарплата: {cand.get('salary', '?')}")
                st.text(f"   🎯 Совместимость: {cand.get('score', 0)}%")
                st.markdown("---")
    else:
        st.warning("⚠️ Не найдено кандидатов, соответствующих критериям")
        
        with st.expander("💡 Как улучшить результаты?"):
            st.markdown(f"""
            **Текущие фильтры:**
            - Минимальный опыт: {final_filters.get('min_experience', 'не задан')} лет
            - Возраст: {final_filters.get('min_age', '?')}-{final_filters.get('max_age', '?')}
            
            **Попробуйте:**
            - Снизить минимальный опыт
            - Расширить возрастные рамки
            - Убрать фильтр по городу
            - Снизить порог IT Threshold
            """)

st.caption("Сделано на Streamlit + SentenceTransformer • AI HR Скринер 2026")