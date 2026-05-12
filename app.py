import streamlit as st
import requests
import pandas as pd

# ==================== НАСТРОЙКИ ====================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI HR Помощник", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ВЫБОР РЕЖИМА ====================
st.title("AI HR Помощник")

st.subheader("Выбор режима работы")
mode = st.radio(
    "Роль пользователя",
    options=["HR-менеджер (поиск сотрудника)", "Соискатель (поиск вакансии)"],
    horizontal=True
)

is_hr = "HR" in mode

if is_hr:
    st.header("Подбор кандидатов под вакансию")
else:
    st.header("Подбор вакансий под резюме")

st.divider()

# ==================== НАСТРОЙКИ ФИЛЬТРАЦИИ ====================
with st.sidebar:
    st.header("Настройки фильтрации")
    
    st.subheader("Источник фильтров")
    
    filter_source = st.radio(
        "Выберите источник",
        options=[
            "Только из текста (автоматически)",
            "Только ручные",
            "И то, и другое"
        ]
    )
    
    use_auto = filter_source in ["Только из текста (автоматически)", "И то, и другое"]
    use_manual = filter_source in ["Только ручные", "И то, и другое"]
    
    st.divider()
    
    if is_hr:
        st.subheader("Требования к кандидату")
    else:
        st.subheader("Ваши данные")
    
    if is_hr:
        col1, col2 = st.columns(2)
        with col1:
            min_age = st.number_input("Мин. возраст", min_value=14, max_value=100, value=None, placeholder="Не указано")
        with col2:
            max_age = st.number_input("Макс. возраст", min_value=14, max_value=100, value=None, placeholder="Не указано")
    else:
        age = st.number_input("Ваш возраст", min_value=14, max_value=100, value=None, placeholder="Не указано")
        min_age = age
        max_age = age
    
    if is_hr:
        min_exp = st.number_input("Мин. опыт кандидата (лет)", min_value=0, max_value=50, value=None, placeholder="Не указано")
    else:
        min_exp = st.number_input("Ваш опыт (лет)", min_value=0, max_value=50, value=None, placeholder="Не указано")
    
    if is_hr:
        city_input = st.text_input("Город работы", value="", placeholder="Оставьте пустым для отключения")
    else:
        city_input = st.text_input("Ваш город", value="", placeholder="Оставьте пустым для отключения")
    
    st.divider()
    
    if is_hr:
        st.subheader("Требуемые навыки и должности")
    else:
        st.subheader("Ваши навыки и желаемая должность")
    
    pos_kw_input = st.text_input("Должность / позиция", value="", placeholder="Python разработчик, Data Scientist, Project Manager")
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    skill_kw_input = st.text_input("Навыки (через запятую)", value="", placeholder="Python, SQL, Docker, FastAPI")
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.divider()
    
    if not is_hr:
        st.subheader("Зарплатные ожидания")
        desired_salary = st.number_input("Желаемая зарплата (тыс. руб.)", min_value=0, max_value=1000, value=None, placeholder="Не указано")
    
    st.divider()
    
    st.subheader("Параметры поиска")
    
    it_threshold = st.slider("IT-порог", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
    min_score = st.slider("Мин. процент совпадения", min_value=0.0, max_value=100.0, value=15.0, step=5.0)
    top_k = st.slider("Топ результатов", min_value=5, max_value=50, value=20, step=5)
    
    use_reranking = st.checkbox("Использовать точное ранжирование (медленнее, но точнее)", value=True)

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================

if is_hr:
    st.subheader("Текст вакансии")
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
        """
    )
    
    uploaded = st.file_uploader("Загрузите файл с резюме (resumes_generated.txt)", type="txt")
    
else:
    st.subheader("Текст вашего резюме")
    vacancy_text = st.text_area(
        "Опишите ваш опыт, навыки и карьерные цели",
        value="",
        height=200,
        placeholder="""
Пример:
Python разработчик, опыт 5 лет.
Навыки: FastAPI, Django, PostgreSQL, Docker, Kubernetes.
Образование: высшее техническое.
Ищу работу в Москве или удаленно.
Возраст: 28 лет.
Ожидаемая зарплата: от 200 000 руб.
        """
    )
    
    uploaded = st.file_uploader("Загрузите файл с вакансиями", type="txt")

st.divider()

# ==================== КНОПКА ЗАПУСКА ====================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_button = st.button("Запустить подбор", type="primary", use_container_width=True)

if run_button:
    
    if not uploaded:
        st.error("Ошибка: файл не загружен")
        st.stop()
    
    if not vacancy_text.strip():
        st.error("Ошибка: текст не введен")
        st.stop()
    
    if is_hr:
        resume_file = ("resumes.txt", uploaded.getvalue(), "text/plain")
    else:
        resume_file = ("vacancies.txt", uploaded.getvalue(), "text/plain")
    
    files = {
        "vacancy_file": ("vacancy.txt", vacancy_text.encode("utf-8"), "text/plain"),
        "resume_file": resume_file
    }
    
    data = {
        "top_k": str(top_k),
        "min_score": str(min_score),
        "it_threshold": str(it_threshold),
        "use_auto_filters": str(use_auto).lower(),
        "use_reranking": str(use_reranking).lower(),
        "mode": "hr" if is_hr else "candidate"
    }
    
    if use_manual:
        if min_age is not None:
            data["min_age"] = str(min_age)
        if max_age is not None:
            data["max_age"] = str(max_age)
        if min_exp is not None:
            data["min_experience"] = str(min_exp)
        if city_input.strip():
            data["city"] = city_input.strip()
        if pos_kw:
            data["position_keywords"] = ",".join(pos_kw)
        if skill_kw:
            data["skills_keywords"] = ",".join(skill_kw)
        if not is_hr and 'desired_salary' in locals() and desired_salary:
            data["desired_salary"] = str(desired_salary)
    
    with st.spinner("Анализ (15-40 секунд)..."):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/match/advanced",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("top_candidates", [])
                
                if candidates:
                    if is_hr:
                        st.success(f"Найдено {len(candidates)} подходящих кандидатов")
                    else:
                        st.success(f"Найдено {len(candidates)} подходящих вакансий")
                    
                    df_data = []
                    for c in candidates:
                        if is_hr:
                            row = {
                                "Имя": c.get("name", "—"),
                                "Возраст": c.get("age", "—"),
                                "Город": c.get("city", "—"),
                                "Опыт": c.get("experience", "—"),
                                "Совпадение %": c.get("match_score", 0),
                                "Точный %": c.get("rerank_score_percent", 0),
                                "Навыки": c.get("skills", "—")[:50]
                            }
                        else:
                            row = {
                                "Должность": c.get("desired_position", "—"),
                                "Компания": c.get("company", "—"),
                                "Город": c.get("city", "—"),
                                "Зарплата": c.get("salary", "—"),
                                "Совпадение %": c.get("match_score", 0),
                                "Точный %": c.get("rerank_score_percent", 0),
                                "Требования": c.get("skills", "—")[:50]
                            }
                        df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Детальная информация")
                    
                    for idx, c in enumerate(candidates):
                        if is_hr:
                            name = c.get("name", "Unknown")
                            title = f"{idx+1}. {name} - Совпадение: {c.get('match_score', 0):.1f}%"
                        else:
                            name = c.get("desired_position", "Unknown")
                            title = f"{idx+1}. {name} - Совпадение: {c.get('match_score', 0):.1f}%"
                        
                        with st.expander(title):
                            col1, col2 = st.columns(2)
                            with col1:
                                if is_hr:
                                    st.write(f"**Имя:** {c.get('name', '—')}")
                                    st.write(f"**Возраст:** {c.get('age', '—')}")
                                    st.write(f"**Опыт:** {c.get('experience', '—')} лет")
                                else:
                                    st.write(f"**Должность:** {c.get('desired_position', '—')}")
                                    st.write(f"**Компания:** {c.get('company', '—')}")
                            with col2:
                                st.write(f"**Город:** {c.get('city', '—')}")
                                st.write(f"**Зарплата:** {c.get('salary', '—')}")
                                st.write(f"**Совпадение:** {c.get('match_score', 0):.1f}%")
                                if c.get('rerank_score_percent'):
                                    st.write(f"**Точное совпадение:** {c.get('rerank_score_percent', 0):.1f}%")
                            
                            if c.get('skills'):
                                skills_str = c.get('skills', '—')
                                if len(skills_str) > 300:
                                    skills_str = skills_str[:300]
                                if is_hr:
                                    st.write(f"**Навыки:** {skills_str}")
                                else:
                                    st.write(f"**Требования:** {skills_str}")
                else:
                    st.warning("Ничего не найдено. Попробуйте снизить порог совпадения.")
            else:
                st.error(f"Ошибка API: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error(f"Не удалось подключиться к бэкенду. Убедитесь, что он запущен на {API_URL}")
        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания. Попробуйте уменьшить количество резюме.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

st.divider()
st.caption("AI HR Помощник | Работает через бэкенд FastAPI")
