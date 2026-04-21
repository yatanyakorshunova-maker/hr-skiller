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
    
    st.subheader("📋 Требования к кандидатам")
    
    min_age = st.number_input("Минимальный возраст", 18, 60, 23, key="min_age_input")
    max_age = st.number_input("Максимальный возраст", 18, 70, 45, key="max_age_input")
    min_exp = st.number_input("Минимальный опыт (лет)", 0, 15, 4, key="min_exp_input")
    
    city_filter = st.text_input("Город (оставь пустым = любой)", "", key="city_input")
    
    position_keywords = st.multiselect(
        "Ключевые слова в должности",
        ["Backend", "Python", "Developer", "Middle", "Senior", "Fullstack", "Engineer", "Team Lead"],
        default=["Backend", "Python", "Developer"],
        key="position_input"
    )
    
    skills_keywords = st.multiselect(
        "Ключевые навыки",
        ["Python", "FastAPI", "Django", "Flask", "PostgreSQL", "Docker", "Kubernetes", "Redis", "SQL", "MongoDB", "Git", "Celery"],
        default=["Python", "FastAPI", "PostgreSQL", "Docker"],
        key="skills_input"
    )
    
    st.divider()
    
    it_threshold = st.slider("IT Threshold (чувствительность)", 0.0, 10.0, 2.0, 0.5, 
                             help="Чем выше значение, тем строже проверка на IT-специальность")
    use_reranking = st.checkbox("Включить reranking (более точный)", value=True)
    min_score = st.slider("Минимальный semantic score (%)", 10, 40, 15,
                          help="Минимальный процент совместимости с вакансией")
    top_k = st.slider("Сколько кандидатов показывать", 5, 50, 20)

# ====================== ОСНОВНАЯ ЧАСТЬ ======================

vacancy_text = st.text_area(
    "📝 Вставь текст вакансии сюда",
    value=st.session_state.vacancy_text,
    height=250,
    key="vacancy_input"
)
st.session_state.vacancy_text = vacancy_text

# Загрузка файла с резюме
uploaded_file = st.file_uploader("📁 Загрузи файл с резюме (resumes_generated.txt)", type=["txt"])

# Кнопка запуска
search_btn = st.button("🚀 Запустить подбор кандидатов", type="primary", use_container_width=True)

# Обработка поиска
if search_btn:
    if uploaded_file is None:
        st.error("❌ Сначала загрузи файл с резюме!")
        st.stop()
    
    # Сохраняем загруженный файл
    with open("resumes_generated.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Файл с резюме загружен!")
    
    # Формируем фильтры (только ручные)
    final_filters = {
        'min_age': min_age,
        'max_age': max_age,
        'min_experience': min_exp,
        'city': city_filter.strip() if city_filter.strip() else None,
        'position_keywords': position_keywords if position_keywords else [],
        'skills_keywords': skills_keywords if skills_keywords else []
    }
    
    # Убираем пустые значения
    final_filters = {k: v for k, v in final_filters.items() if v}
    
    # Показываем примененные фильтры
    st.subheader("🔍 Примененные фильтры:")
    if final_filters:
        st.json(final_filters)
    else:
        st.info("Фильтры не применены")
    
    # Запуск анализа
    with st.spinner("📊 Анализируем кандидатов... (20-60 секунд)"):
        resumes = hr_core.load_resumes_from_file("resumes_generated.txt")
        original_count = len(resumes)
        
        # Диагностика загруженных резюме
        st.write("---")
        st.subheader("📋 Загруженные резюме:")
        for i, r in enumerate(resumes[:5]):
            age_val = extract_number_from_string(r.get('age'))
            exp_val = extract_number_from_string(r.get('experience'))
            st.text(f"{i+1}. {r.get('name')} | Возраст: {r.get('age')} → {age_val} | Опыт: {r.get('experience')} → {exp_val} | Город: {r.get('city')}")
        
        # Фильтруем кандидатов
        filtered_candidates = []
        rejected_candidates = []
        
        for resume in resumes:
            # Извлекаем возраст и опыт как числа
            age = extract_number_from_string(resume.get('age'))
            experience = extract_number_from_string(resume.get('experience'))
            
            # Проверяем фильтры
            passed = True
            reason = None
            
            # Проверка имени
            name = resume.get('name', '')
            if not name or len(name) < 3:
                passed = False
                reason = "Некорректное имя"
            
            # Проверка города
            elif city_filter and city_filter.strip():
                city_norm = normalize_city(resume.get('city', ''))
                if city_filter.lower() not in city_norm.lower():
                    passed = False
                    reason = f"Город '{resume.get('city')}' ≠ '{city_filter}'"
            
            # Проверка возраста
            elif min_age and age and age < min_age:
                passed = False
                reason = f"Возраст {age} < {min_age}"
            elif max_age and age and age > max_age:
                passed = False
                reason = f"Возраст {age} > {max_age}"
            
            # Проверка опыта
            elif min_exp and experience and experience < min_exp:
                passed = False
                reason = f"Опыт {experience} < {min_exp}"
            
            # Проверка должности
            elif position_keywords and len(position_keywords) > 0:
                pos_text = resume.get('desired_position', '').lower()
                if not any(kw.lower() in pos_text for kw in position_keywords):
                    passed = False
                    reason = f"Должность не содержит ключевых слов: {position_keywords}"
            
            # Проверка навыков
            elif skills_keywords and len(skills_keywords) > 0:
                skills_text = resume.get('skills', '').lower()
                if not any(kw.lower() in skills_text for kw in skills_keywords):
                    passed = False
                    reason = f"Навыки не содержат ключевых слов: {skills_keywords}"
            
            # IT проверка
            else:
                is_it, it_reason = is_it_candidate(resume, threshold=it_threshold)
                if not is_it:
                    passed = False
                    reason = f"Не IT-специальность: {it_reason}"
            
            if passed:
                # Добавляем parsed поля для hr_core
                resume['parsed_age'] = age
                resume['parsed_experience'] = experience
                filtered_candidates.append(resume)
            else:
                rejected_candidates.append({
                    'name': resume.get('name', '?')[:35],
                    'age': age if age else '?',
                    'experience': experience if experience else '?',
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
            st.info("✅ Нет отсеянных кандидатов")
    
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
                if cand.get('rerank_score_percent'):
                    st.text(f"   🎯 Точность (rerank): {cand.get('rerank_score_percent')}%")
                st.markdown("---")
    else:
        st.warning("⚠️ Не найдено кандидатов, соответствующих критериям")
        
        with st.expander("💡 Как улучшить результаты?"):
            st.markdown(f"""
            **Текущие фильтры:**
            - Минимальный опыт: {min_exp} лет
            - Возраст: {min_age}-{max_age} лет
            - Город: {city_filter if city_filter else 'любой'}
            - Ключевые слова должности: {', '.join(position_keywords) if position_keywords else 'любые'}
            - Ключевые навыки: {', '.join(skills_keywords) if skills_keywords else 'любые'}
            
            **Попробуйте:**
            - Снизить минимальный опыт
            - Расширить возрастные рамки
            - Убрать фильтр по городу
            - Добавить больше ключевых слов
            - Снизить порог IT Threshold
            - Уменьшить минимальный semantic score
            """)

st.caption("Сделано на Streamlit + SentenceTransformer • AI HR Скринер 2026")