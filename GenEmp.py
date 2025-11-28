import json
import random
import requests
import networkx as nx
from pyvis.network import Network
from dataclasses import dataclass, field
from typing import List, Set, Dict
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import IPython # Explicit import for the fix

# ==========================================
# 1. AUTO-SEEDER
# ==========================================
class AutoSeeder:
    DOMAIN_KEYWORDS = {
        "Bio": {"life", "cell", "organic", "body", "nature", "living", "genetic", "viral", "evolution", "species"},
        "Tech": {"computer", "digital", "silicon", "electronic", "code", "data", "cyber", "machine", "robot", "network"},
        "Physics": {"energy", "matter", "space", "time", "force", "quantum", "gravity", "light", "particle", "chaos", "thermodynamics"},
        "Social": {"human", "society", "people", "culture", "politics", "law", "community", "mind", "group", "consciousness", "history"},
        "Phil": {"truth", "logic", "reason", "moral", "abstract", "thought", "existential", "meaning", "philosophy"},
        "Abstract": {"math", "number", "geometry", "fractal", "recursion", "pattern", "infinite", "set", "theory"},
        "General": {"object", "thing", "generic"}
    }

    @staticmethod
    def fetch_data(concept_name: str):
        props = set()
        try:
            for rel in ["rel_jjb", "rel_trg"]:
                url = f"https://api.datamuse.com/words?{rel}={concept_name}&max=5"
                data = requests.get(url).json()
                for item in data:
                    props.add(item['word'])
        except:
            return set(), "General"
        return props

    @staticmethod
    def guess_domain(props: Set[str]) -> str:
        scores = {k: 0 for k in AutoSeeder.DOMAIN_KEYWORDS}
        for p in props:
            for domain, keywords in AutoSeeder.DOMAIN_KEYWORDS.items():
                if any(k in p for k in keywords):
                    scores[domain] += 1
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "General"

# ==========================================
# 2. CORE LOGIC
# ==========================================
@dataclass
class Concept:
    name: str
    domain: str
    properties: Set[str]
    dimension: int = 1
    id: str = field(init=False)
    def __post_init__(self): self.id = self.name
    def to_dict(self): return {"name": self.name, "domain": self.domain, "properties": list(self.properties), "dimension": self.dimension}

def op_scale_up(c): return Concept(f"Global {c.name}", c.domain, c.properties.union({"systemic"}), c.dimension)
def op_invert(c): return Concept(f"Anti-{c.name}", c.domain, c.properties.union({"inverted"}), c.dimension)
def op_lift(c): return Concept(f"Meta-{c.name}", "Abstract", c.properties.union({"abstract"}), c.dimension + 1)
def op_recurse(c): return Concept(f"Self-{c.name}", c.domain, c.properties.union({"recursive", "loop"}), c.dimension)
def op_rotate(c, all_domains):
    available = [d for d in all_domains if d != c.domain]
    new_dom = random.choice(available) if available else "General"
    return Concept(f"{c.name}-in-{new_dom}", new_dom, c.properties.union({f"applied-to-{new_dom}"}), c.dimension)
def op_blend(c1, c2):
    dom = f"{c1.domain}-{c2.domain}" if c1.domain != c2.domain else c1.domain
    n1, n2 = c1.name.split(' ')[0], c2.name.split(' ')[-1]
    return Concept(f"{n1}-{n2} Hybrid", dom, c1.properties.union(c2.properties), max(c1.dimension, c2.dimension))

# ==========================================
# 3. VISUALIZER
# ==========================================
class VisualGEE:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts = {}
        self.unary_ops = [op_scale_up, op_invert, op_lift, op_recurse]
        self.colors = {"Bio": "#97c2fc", "Tech": "#ffff00", "Social": "#fb7e81", "Physics": "#eb7df4", "Abstract": "#7be141", "Phil": "#ffa500", "General": "#e2e2e2"}

    def add_concept(self, c: Concept):
        if c.name not in self.concepts:
            self.concepts[c.name] = c
            color = self.colors.get(c.domain, "#c2f5ff") 
            self.graph.add_node(c.name, label=c.name, title=f"Props: {list(c.properties)}", color=color, size=15+(c.dimension*5))

    def run(self, seeds: List[Concept], all_domains: List[str], steps=2):
        frontier = seeds
        for s in seeds: self.add_concept(s)
        for i in range(steps):
            next_frontier = []
            for c in frontier:
                ro = op_rotate(c, all_domains)
                if ro.name not in self.concepts:
                    self.add_concept(ro)
                    self.graph.add_edge(c.name, ro.name, title="rotate")
                    next_frontier.append(ro)

                for op in self.unary_ops:
                    new_c = op(c)
                    if new_c.name not in self.concepts:
                        self.add_concept(new_c)
                        self.graph.add_edge(c.name, new_c.name, title=op.__name__[3:])
                        next_frontier.append(new_c)
                if len(self.concepts) > 1:
                    partner = random.choice(list(self.concepts.values()))
                    if partner.name != c.name:
                        blend = op_blend(c, partner)
                        if blend.name not in self.concepts:
                            self.add_concept(blend)
                            self.graph.add_edge(c.name, blend.name, title="blend")
                            self.graph.add_edge(partner.name, blend.name, title="blend")
                            next_frontier.append(blend)
            frontier = next_frontier

    def save_viz(self):
        net = Network(height="650px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote', select_menu=True)
        net.from_nx(self.graph)
        net.show_buttons(filter_=['physics'])
        return net.save_graph("concept_graph.html")

# ==========================================
# 4. DASHBOARD v0.9 (The Fix!)
# ==========================================
class GEEDashboard:
    def __init__(self):
        self.seeds = []
        self.domains = ["Auto-Detect", "Social", "Tech", "Bio", "Physics", "Phil", "Abstract", "General"]
        
        self.w_name = widgets.Text(placeholder='Concept (Press Enter)', description='Name:', layout=widgets.Layout(width='200px'))
        self.w_domain = widgets.Dropdown(options=self.domains, description='Domain:', layout=widgets.Layout(width='200px'))
        self.w_props = widgets.Text(placeholder='Props (Optional)', description='Props:', layout=widgets.Layout(width='300px'))
        self.w_add_btn = widgets.Button(description="Add", button_style='info', icon='plus', layout=widgets.Layout(width='80px'))
        
        self.w_new_dom = widgets.Text(placeholder='New Domain...', layout=widgets.Layout(width='150px'))
        self.w_add_dom_btn = widgets.Button(description="Add Domain", icon='plus-circle', layout=widgets.Layout(width='120px'))
        self.w_run_btn = widgets.Button(description=" GENERATE GALAXY", button_style='success', icon='rocket', layout=widgets.Layout(width='98%', height='50px'))
        self.w_steps = widgets.IntSlider(value=2, min=1, max=3, description='Steps:')
        
        self.out_seed_list = widgets.Output()
        self.out_viz = widgets.Output()

        # Events
        self.w_add_btn.on_click(self.add_seed)
        self.w_name.on_submit(self.add_seed)  # ENTER KEY MAGIC
        self.w_props.on_submit(self.add_seed) # ENTER KEY MAGIC
        
        self.w_add_dom_btn.on_click(self.add_custom_domain)
        self.w_run_btn.on_click(self.run_engine)

        self.input_box = widgets.HBox([self.w_name, self.w_domain, self.w_props, self.w_add_btn])
        self.domain_box = widgets.HBox([widgets.Label(value="Custom Domain:"), self.w_new_dom, self.w_add_dom_btn])
        
        display(widgets.VBox([
            widgets.HTML("<h2>🧠 GEE v0.9: Richard's Fix</h2>"),
            self.input_box,
            self.domain_box,
            widgets.HTML("<hr>"),
            widgets.Label("Active Seeds:"),
            self.out_seed_list,
            widgets.HTML("<hr>"),
            widgets.HBox([self.w_steps]),
            self.w_run_btn,
            self.out_viz
        ]))
        self.render_seed_list()

    def add_custom_domain(self, b):
        dom = self.w_new_dom.value.strip()
        if dom and dom not in self.domains:
            self.domains.append(dom)
            self.w_domain.options = self.domains
            self.w_domain.value = dom
            self.w_new_dom.value = ""

    def add_seed(self, b):
        name = self.w_name.value.strip()
        if not name: return
        
        domain_choice = self.w_domain.value
        user_props = self.w_props.value
        
        if not user_props.strip():
            props = AutoSeeder.fetch_data(name)
        else:
            props = {p.strip() for p in user_props.split(',') if p.strip()}
            
        if domain_choice == "Auto-Detect":
            final_domain = AutoSeeder.guess_domain(props)
        else:
            final_domain = domain_choice

        new_seed = Concept(name, final_domain, props)
        self.seeds.append(new_seed)
        
        # Clear inputs for speed entry
        self.w_name.value = ""
        self.w_props.value = ""
        self.render_seed_list()

    def delete_seed(self, b):
        idx = int(b.tooltip)
        if 0 <= idx < len(self.seeds):
            del self.seeds[idx]
            self.render_seed_list()

    def render_seed_list(self):
        with self.out_seed_list:
            clear_output()
            rows = []
            for i, s in enumerate(self.seeds):
                btn_del = widgets.Button(description="X", button_style='danger', layout=widgets.Layout(width='30px', height='25px'), tooltip=str(i))
                btn_del.on_click(self.delete_seed)
                lbl_info = widgets.Label(value=f" {s.name} [{s.domain}] -- {list(s.properties)[:4]}...")
                rows.append(widgets.HBox([btn_del, lbl_info]))
            
            if not rows: print("No seeds yet.")
            else: display(widgets.VBox(rows))

    def run_engine(self, b):
        if not self.seeds: return
        with self.out_viz:
            clear_output()
            print("Igniting Engine...")
            viz = VisualGEE()
            viz.run(self.seeds, self.domains, steps=self.w_steps.value)
            path = viz.save_viz()
            
            # THE FIX: Explicit IPython display call
            IPython.display.display(IPython.display.HTML(filename=path))

dashboard = GEEDashboard()