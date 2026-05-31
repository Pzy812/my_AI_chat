"""Neo4j 连接示例与测试脚本。

生产环境请使用 neo4j_store.py（GraphRAG 社区版：每次上传清空 neo4j 库并重建图谱）。
连接参数可通过环境变量 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 配置。
"""
from neo4j_store import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    close_driver,
    get_driver,
    list_session_modules,
)
# ======================
# 以下为演示数据（可选运行）
# ======================
from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ======================
# 2. 清空旧数据
# ======================
def clear_data():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("已清空所有旧数据")

# ======================
# 3. 创建更多 Person 节点
# ======================
def create_persons():
    persons = [
        {"name": "张三", "age": 20, "city": "北京"},
        {"name": "李四", "age": 21, "city": "上海"},
        {"name": "王五", "age": 22, "city": "深圳"},
        {"name": "赵六", "age": 23, "city": "广州"},
        {"name": "陈七", "age": 25, "city": "北京"},
        {"name": "杨八", "age": 28, "city": "上海"},
        {"name": "吴九", "age": 30, "city": "深圳"},
        {"name": "郑十", "age": 27, "city": "广州"},
        {"name": "林一", "age": 24, "city": "北京"},
        {"name": "周二", "age": 29, "city": "上海"},
    ]
    with driver.session() as session:
        session.run("""
            UNWIND $persons AS p
            MERGE (person:Person {name: p.name})
            SET person.age = p.age, person.city = p.city
        """, persons=persons)
    print(f"✅ 创建/更新 {len(persons)} 个人")

# ======================
# 4. 创建 City 节点
# ======================
def create_cities():
    cities = ["北京", "上海", "深圳", "广州"]
    with driver.session() as session:
        session.run("""
            UNWIND $cities AS c
            MERGE (city:City {name: c})
        """, cities=cities)
    print(f"✅ 创建 {len(cities)} 个城市")

# ======================
# 5. 创建 Company 节点
# ======================
def create_companies():
    companies = [
        {"name": "阿里", "city": "杭州"},
        {"name": "腾讯", "city": "深圳"},
        {"name": "百度", "city": "北京"}
    ]
    with driver.session() as session:
        session.run("""
            UNWIND $companies AS comp
            MERGE (c:Company {name: comp.name})
            SET c.city = comp.city
        """, companies=companies)
    print(f"✅ 创建 {len(companies)} 个公司")

# ======================
# 6. 创建关系：FRIEND / LIVE_IN / WORK_AT
# ======================
def create_relations():
    friends = [
        ("张三", "李四"),
        ("李四", "王五"),
        ("王五", "赵六"),
        ("张三", "陈七"),
        ("陈七", "林一"),
        ("杨八", "周二"),
        ("吴九", "郑十")
    ]
    lives_in = [
        ("张三", "北京"),
        ("李四", "上海"),
        ("王五", "深圳"),
        ("赵六", "广州"),
        ("陈七", "北京"),
        ("杨八", "上海"),
        ("吴九", "深圳"),
        ("郑十", "广州"),
        ("林一", "北京"),
        ("周二", "上海")
    ]
    works_at = [
        ("张三", "百度"),
        ("李四", "腾讯"),
        ("王五", "腾讯"),
        ("陈七", "阿里"),
        ("杨八", "百度")
    ]

    with driver.session() as session:
        # 朋友关系
        session.run("""
            UNWIND $pairs AS pair
            MATCH (a:Person{name: pair[0]}), (b:Person{name: pair[1]})
            MERGE (a)-[:FRIEND]->(b)
        """, pairs=friends)

        # 住在
        session.run("""
            UNWIND $pairs AS pair
            MATCH (p:Person{name: pair[0]}), (c:City{name: pair[1]})
            MERGE (p)-[:LIVE_IN]->(c)
        """, pairs=lives_in)

        # 工作在
        session.run("""
            UNWIND $pairs AS pair
            MATCH (p:Person{name: pair[0]}), (comp:Company{name: pair[1]})
            MERGE (p)-[:WORK_AT]->(comp)
        """, pairs=works_at)

    print("✅ 所有关系创建完成")

# ======================
# 7. 查询：所有人
# ======================
def query_all_persons():
    print("\n===== 所有人 =====")
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Person)
            RETURN p.name AS name, p.age AS age, p.city AS city
            ORDER BY age
        """)
        for rec in res:
            print(f"{rec['name']} 年龄:{rec['age']} 城市:{rec['city']}")

# ======================
# 8. 查询：张三的朋友链（最多3层）
# ======================
def query_friend_path():
    print("\n===== 张三的朋友链 =====")
    with driver.session() as session:
        res = session.run("""
            MATCH (a:Person{name:'张三'})-[:FRIEND*1..3]->(f:Person)
            RETURN DISTINCT f.name AS name
        """)
        for rec in res:
            print("→", rec["name"])

# ======================
# 9. 查询：在腾讯工作的人
# ======================
def query_company_people():
    print("\n===== 在腾讯工作的人 =====")
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Person)-[:WORK_AT]->(:Company{name:'腾讯'})
            RETURN p.name
        """)
        for rec in res:
            print(rec["p.name"])

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # clear_data()
    create_cities()
    create_companies()
    create_persons()
    create_relations()

    query_all_persons()
    query_friend_path()
    query_company_people()

    driver.close()
    print("\n🎉 全部执行完成！")