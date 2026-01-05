import networkx as nx
import random
import numpy as np
import math
from scipy.spatial import Delaunay
import networkx as nx

names = [
    "James", # 100 most common male and female US names over the last centruy (https://www.ssa.gov/oact/babynames/decades/century.html)
    "Michael",
    "Robert",
    "John",
    "David",
    "William",
    "Richard",
    "Joseph",
    "Thomas",
    "Christopher",
    "Charles",
    "Daniel",
    "Matthew",
    "Anthony",
    "Mark",
    "Donald",
    "Steven",
    "Andrew",
    "Paul",
    "Joshua",
    "Kenneth",
    "Kevin",
    "Brian",
    "Timothy",
    "Ronald",
    "George",
    "Jason",
    "Edward",
    "Jeffrey",
    "Ryan",
    "Jacob",
    "Nicholas",
    "Gary",
    "Eric",
    "Jonathan",
    "Stephen",
    "Larry",
    "Justin",
    "Scott",
    "Brandon",
    "Benjamin",
    "Samuel",
    "Gregory",
    "Alexander",
    "Patrick",
    "Frank",
    "Raymond",
    "Jack",
    "Dennis",
    "Jerry",
    "Tyler",
    "Aaron",
    "Jose",
    "Adam",
    "Nathan",
    "Henry",
    "Zachary",
    "Douglas",
    "Peter",
    "Kyle",
    "Noah",
    "Ethan",
    "Jeremy",
    "Christian",
    "Walter",
    "Keith",
    "Austin",
    "Roger",
    "Terry",
    "Sean",
    "Gerald",
    "Carl",
    "Dylan",
    "Harold",
    "Jordan",
    "Jesse",
    "Bryan",
    "Lawrence",
    "Arthur",
    "Gabriel",
    "Bruce",
    "Logan",
    "Billy",
    "Joe",
    "Alan",
    "Juan",
    "Elijah",
    "Willie",
    "Albert",
    "Wayne",
    "Randy",
    "Mason",
    "Vincent",
    "Liam",
    "Roy",
    "Bobby",
    "Caleb",
    "Bradley",
    "Russell",
    "Lucas",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Karen",
    "Sarah",
    "Lisa",
    "Nancy",
    "Sandra",
    "Betty",
    "Ashley",
    "Emily",
    "Kimberly",
    "Margaret",
    "Donna",
    "Michelle",
    "Carol",
    "Amanda",
    "Melissa",
    "Deborah",
    "Stephanie",
    "Rebecca",
    "Sharon",
    "Laura",
    "Cynthia",
    "Dorothy",
    "Amy",
    "Kathleen",
    "Angela",
    "Shirley",
    "Emma",
    "Brenda",
    "Pamela",
    "Nicole",
    "Anna",
    "Samantha",
    "Katherine",
    "Christine",
    "Debra",
    "Rachel",
    "Carolyn",
    "Janet",
    "Maria",
    "Olivia",
    "Heather",
    "Helen",
    "Catherine",
    "Diane",
    "Julie",
    "Victoria",
    "Joyce",
    "Lauren",
    "Kelly",
    "Christina",
    "Ruth",
    "Joan",
    "Virginia",
    "Judith",
    "Evelyn",
    "Hannah",
    "Andrea",
    "Megan",
    "Cheryl",
    "Jacqueline",
    "Madison",
    "Teresa",
    "Abigail",
    "Sophia",
    "Martha",
    "Sara",
    "Gloria",
    "Janice",
    "Kathryn",
    "Ann",
    "Isabella",
    "Judy",
    "Charlotte",
    "Julia",
    "Grace",
    "Amber",
    "Alice",
    "Jean",
    "Denise",
    "Frances",
    "Danielle",
    "Marilyn",
    "Natalie",
    "Beverly",
    "Diana",
    "Brittany",
    "Theresa",
    "Kayla",
    "Alexis",
    "Doris",
    "Lori",
    "Tiffany",
]

def relabel_and_name_vertices(graph: nx.Graph) -> nx.Graph:
    graph = nx.relabel_nodes(graph, {v: i for i, v in enumerate(graph.nodes)})
    node_names = random.sample(names, graph.order())
    nx.set_node_attributes(graph, {i: name for i, name in enumerate(node_names)}, "name")
    return graph

def _edges_to_graph(n, edges, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

def generate_chain_graph(n, seed=None, directed=True):
    """0->1->2->...->n-1"""
    edges = [(i, i + 1) for i in range(n - 1)]
    return _edges_to_graph(n, edges, directed)

def generate_star_graph(n, seed=None, directed=True, center=0):
    """center->others 的出星型"""
    edges = [(center, i) for i in range(n) if i != center]
    return _edges_to_graph(n, edges, directed)

def generate_tree_graph(n, seed=None, directed=True):
    """完全二叉树样式的前馈有向树"""
    edges = []
    i = 0
    while True:
        c1, c2 = 2 * i + 1, 2 * i + 2
        if c1 >= n:
            break
        edges.append((i, c1))
        if c2 < n:
            edges.append((i, c2))
        i += 1
    return _edges_to_graph(n, edges, directed)

def generate_net_graph(n, seed=None, directed=True):
    """完全 DAG：所有 u<v 的有向边"""
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    return _edges_to_graph(n, edges, directed)

def generate_mlp_graph(n, seed=None, directed=True):
    """分层前馈：L=floor(log2 n)，第一层吸收余数；层间全连接"""
    L = max(1, int(math.log(n, 2)))
    sizes = [n // L for _ in range(L)]
    sizes[0] += n % L
    # 层的索引区间
    bounds = []
    start = 0
    for sz in sizes:
        bounds.append((start, start + sz))
        start += sz
    edges = []
    for (a0, a1), (b0, b1) in zip(bounds[:-1], bounds[1:]):
        for u in range(a0, a1):
            for v in range(b0, b1):
                edges.append((u, v))
    return _edges_to_graph(n, edges, directed)

def generate_random_dag(n, seed=None, directed=True):
    """随机 DAG：从所有 u<v 中随机取边，边数∈[n-1, n(n-1)/2]"""
    rng = random.Random(seed)
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_min, m_max = n - 1, n * (n - 1) // 2
    m = rng.randint(m_min, m_max)
    rng.shuffle(all_pairs)
    edges = all_pairs[:m]
    return _edges_to_graph(n, edges, directed)