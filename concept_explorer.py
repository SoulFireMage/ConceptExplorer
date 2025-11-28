import json
import random
import requests
import networkx as nx
from pyvis.network import Network
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_provider import get_provider, get_available_providers, LLMProvider
from semantic_resolver import SemanticResolver, ResolvedConcept

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

    # Reusable session for connection pooling
    _session = None

    @classmethod
    def get_session(cls):
        if cls._session is None:
            cls._session = requests.Session()
        return cls._session

    @classmethod
    def fetch_data(cls, concept_name: str):
        props = set()
        session = cls.get_session()
        try:
            for rel in ["rel_jjb", "rel_trg"]:
                url = f"https://api.datamuse.com/words?{rel}={concept_name}&max=5"
                data = session.get(url, timeout=5).json()
                for item in data:
                    props.add(item['word'])
        except Exception:
            return set()
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
    # Avoid self-blending
    if c1.name == c2.name:
        return None

    dom = f"{c1.domain}-{c2.domain}" if c1.domain != c2.domain else c1.domain

    # Strip prefixes to find the root concept name
    def strip_prefixes(n):
        for prefix in ["Global ", "Anti-", "Meta-", "Self-"]:
            n = n.replace(prefix, "")
        # Also strip "-in-Domain" suffixes and "Hybrid" suffix
        if "-in-" in n:
            n = n.split("-in-")[0]
        n = n.replace(" Hybrid", "")
        return n.strip()

    n1 = strip_prefixes(c1.name)
    n2 = strip_prefixes(c2.name)

    # Check if they are actually the same root
    if n1 == n2:
        return None

    new_name = f"{n1}-{n2} Hybrid"
    return Concept(new_name, dom, c1.properties.union(c2.properties), max(c1.dimension, c2.dimension))

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

    def run(self, seeds: List[Concept], all_domains: List[str], steps=2, max_nodes=100, max_frontier=20):
        frontier = seeds
        for s in seeds:
            self.add_concept(s)

        for i in range(steps):
            # Stop if we've hit the node limit
            if len(self.concepts) >= max_nodes:
                break

            next_frontier = []
            for c in frontier:
                # Check node limit before each operation
                if len(self.concepts) >= max_nodes:
                    break

                ro = op_rotate(c, all_domains)
                if ro.name not in self.concepts:
                    self.add_concept(ro)
                    self.graph.add_edge(c.name, ro.name, title="rotate")
                    next_frontier.append(ro)

                for op in self.unary_ops:
                    if len(self.concepts) >= max_nodes:
                        break
                    new_c = op(c)
                    if new_c.name not in self.concepts:
                        self.add_concept(new_c)
                        self.graph.add_edge(c.name, new_c.name, title=op.__name__[3:])
                        next_frontier.append(new_c)

                if len(self.concepts) > 1 and len(self.concepts) < max_nodes:
                    partner = random.choice(list(self.concepts.values()))
                    if partner.name != c.name:
                        blend = op_blend(c, partner)
                        # op_blend now returns None for invalid blends
                        if blend is not None and blend.name not in self.concepts:
                            self.add_concept(blend)
                            self.graph.add_edge(c.name, blend.name, title="blend")
                            self.graph.add_edge(partner.name, blend.name, title="blend")
                            next_frontier.append(blend)

            # Prune frontier if it's too large to prevent exponential explosion
            if len(next_frontier) > max_frontier:
                frontier = random.sample(next_frontier, max_frontier)
            else:
                frontier = next_frontier

    def save_viz(self, filename="concept_graph.html"):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote', select_menu=True)

        # Apply physics settings to stabilize the graph layout
        net.set_options("""
        {
          "nodes": {
            "font": { "size": 16, "face": "tahoma" },
            "borderWidth": 2
          },
          "edges": {
            "color": { "inherit": true },
            "smooth": { "type": "continuous" }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)

        net.from_nx(self.graph)
        net.save_graph(filename)
        return filename

# ==========================================
# 4. FLASK WEB APP
# ==========================================
app = Flask(__name__)

# Global storage for seeds
seeds = []
domains = ["Auto-Detect", "Social", "Tech", "Bio", "Physics", "Phil", "Abstract", "General"]

# Store the last generated graph for resolution
last_generated_viz: Optional[VisualGEE] = None
resolved_concepts: Dict[str, ResolvedConcept] = {}  # Cache of resolved concepts
show_discarded = True  # Toggle for showing/hiding discarded concepts

# ==========================================
# LLM Settings Management
# ==========================================
class LLMSettings:
    """Manages LLM provider configuration"""

    def __init__(self):
        self.provider_name = os.getenv("DEFAULT_PROVIDER", "gemini")
        self.model = os.getenv("DEFAULT_MODEL", "")
        self._provider: Optional[LLMProvider] = None

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider from environment"""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "lmstudio": None  # LM Studio doesn't need a key
        }
        env_var = key_map.get(provider_name)
        if env_var is None:
            return "lm-studio"  # Placeholder for LM Studio
        return os.getenv(env_var)

    def get_provider(self) -> Optional[LLMProvider]:
        """Get the currently configured LLM provider"""
        api_key = self.get_api_key(self.provider_name)

        kwargs = {}
        if self.provider_name == "lmstudio":
            kwargs["base_url"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

        model = self.model if self.model else None

        try:
            return get_provider(self.provider_name, api_key=api_key, model=model, **kwargs)
        except Exception as e:
            print(f"Error creating provider: {e}")
            return None

    def update(self, provider_name: str, model: str = ""):
        """Update the current settings"""
        self.provider_name = provider_name
        self.model = model
        self._provider = None  # Reset cached provider

    def get_status(self) -> dict:
        """Get current configuration status"""
        providers_info = get_available_providers()
        configured = {}

        for name in providers_info:
            api_key = self.get_api_key(name)
            if name == "lmstudio":
                # Check if LM Studio is running
                try:
                    provider = get_provider(name, base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
                    configured[name] = provider.is_configured()
                except Exception:
                    configured[name] = False
            else:
                configured[name] = bool(api_key and len(api_key) > 0)

        return {
            "current_provider": self.provider_name,
            "current_model": self.model,
            "providers": providers_info,
            "configured": configured
        }


# Global LLM settings instance
llm_settings = LLMSettings()

@app.route('/')
def index():
    return render_template('index.html', domains=domains)

@app.route('/add_seed', methods=['POST'])
def add_seed():
    data = request.json
    name = data.get('name', '').strip()
    domain_choice = data.get('domain', 'Auto-Detect')
    user_props = data.get('properties', '').strip()

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    # Get properties
    if not user_props:
        props = AutoSeeder.fetch_data(name)
    else:
        props = {p.strip() for p in user_props.split(',') if p.strip()}

    # Determine domain
    if domain_choice == "Auto-Detect":
        final_domain = AutoSeeder.guess_domain(props)
    else:
        final_domain = domain_choice

    new_seed = Concept(name, final_domain, props)
    seeds.append(new_seed)

    return jsonify({
        'success': True,
        'seed': {
            'name': new_seed.name,
            'domain': new_seed.domain,
            'properties': list(new_seed.properties)
        }
    })

@app.route('/delete_seed/<int:index>', methods=['DELETE'])
def delete_seed(index):
    if 0 <= index < len(seeds):
        del seeds[index]
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid index'}), 400

@app.route('/get_seeds', methods=['GET'])
def get_seeds():
    return jsonify([s.to_dict() for s in seeds])

@app.route('/add_domain', methods=['POST'])
def add_domain():
    data = request.json
    domain = data.get('domain', '').strip()
    if domain and domain not in domains:
        domains.append(domain)
        return jsonify({'success': True, 'domains': domains})
    return jsonify({'error': 'Invalid domain'}), 400

@app.route('/generate', methods=['POST'])
def generate():
    global last_generated_viz, resolved_concepts

    data = request.json
    steps = data.get('steps', 2)

    if not seeds:
        return jsonify({'error': 'No seeds added'}), 400

    viz = VisualGEE()
    viz.run(seeds, domains, steps=steps)
    viz.save_viz('static/concept_graph.html')

    # Store for later resolution
    last_generated_viz = viz
    resolved_concepts.clear()  # Clear previous resolutions

    return jsonify({
        'success': True,
        'url': '/view',
        'concept_count': len(viz.concepts)
    })

@app.route('/view')
def view():
    return send_file('static/concept_graph.html')

@app.route('/clear', methods=['POST'])
def clear():
    seeds.clear()
    return jsonify({'success': True})

# ==========================================
# LLM Settings Endpoints
# ==========================================
@app.route('/api/llm/status', methods=['GET'])
def get_llm_status():
    """Get current LLM configuration status"""
    return jsonify(llm_settings.get_status())

@app.route('/api/llm/settings', methods=['POST'])
def update_llm_settings():
    """Update LLM provider and model settings"""
    data = request.json
    provider = data.get('provider', llm_settings.provider_name)
    model = data.get('model', '')

    # Validate provider
    available = get_available_providers()
    if provider not in available:
        return jsonify({'error': f'Unknown provider: {provider}'}), 400

    # Validate model if specified
    if model and model not in available[provider]['models']:
        return jsonify({'error': f'Unknown model: {model} for provider {provider}'}), 400

    llm_settings.update(provider, model)

    return jsonify({
        'success': True,
        'provider': provider,
        'model': model or available[provider]['default_model']
    })

@app.route('/api/llm/test', methods=['POST'])
def test_llm():
    """Test the current LLM configuration with a simple prompt"""
    provider = llm_settings.get_provider()

    if provider is None:
        return jsonify({'error': 'No LLM provider configured'}), 400

    if not provider.is_configured():
        return jsonify({'error': f'Provider {llm_settings.provider_name} is not configured (missing API key or server not running)'}), 400

    try:
        response = provider.complete(
            system_prompt="You are a helpful assistant. Respond in exactly 10 words or less.",
            user_prompt="Say hello and confirm you're working."
        )
        return jsonify({
            'success': True,
            'response': response.content,
            'model': response.model
        })
    except Exception as e:
        return jsonify({'error': f'LLM test failed: {str(e)}'}), 500

# ==========================================
# Semantic Resolver Endpoints
# ==========================================
@app.route('/api/resolve/status', methods=['GET'])
def get_resolve_status():
    """Get current resolution status"""
    if last_generated_viz is None:
        return jsonify({
            'has_graph': False,
            'total_concepts': 0,
            'resolved_count': 0,
            'kept_count': 0,
            'discarded_count': 0
        })

    kept = sum(1 for r in resolved_concepts.values() if r.status == "KEEP")
    discarded = sum(1 for r in resolved_concepts.values() if r.status == "DISCARD")

    return jsonify({
        'has_graph': True,
        'total_concepts': len(last_generated_viz.concepts),
        'resolved_count': len(resolved_concepts),
        'kept_count': kept,
        'discarded_count': discarded,
        'show_discarded': show_discarded
    })

@app.route('/api/resolve/crystallize', methods=['POST'])
def crystallize_concepts():
    """Resolve all concepts in the current graph using the LLM"""
    global resolved_concepts

    if last_generated_viz is None:
        return jsonify({'error': 'No graph generated yet. Generate a concept galaxy first.'}), 400

    provider = llm_settings.get_provider()
    if provider is None:
        return jsonify({'error': 'No LLM provider configured'}), 400

    if not provider.is_configured():
        return jsonify({'error': f'Provider {llm_settings.provider_name} is not configured'}), 400

    resolver = SemanticResolver(provider)

    # Get concepts that haven't been resolved yet
    concepts_to_resolve = [
        {"name": name, "domain": concept.domain}
        for name, concept in last_generated_viz.concepts.items()
        if name not in resolved_concepts
    ]

    if not concepts_to_resolve:
        return jsonify({
            'success': True,
            'message': 'All concepts already resolved',
            'resolved': []
        })

    # Resolve using batched LLM calls (5 concepts per request for ~80% speedup)
    try:
        batch_results = resolver.resolve_batch(concepts_to_resolve, use_batching=True, batch_size=5)

        results = []
        errors = []

        for result in batch_results:
            resolved_concepts[result.original] = result
            results.append(result.to_dict())
            if result.error:
                errors.append({"concept": result.original, "error": result.error})

    except Exception as e:
        return jsonify({'error': f'Batch resolution failed: {str(e)}'}), 500

    # Regenerate the visualization with resolved names
    _regenerate_resolved_graph()

    return jsonify({
        'success': True,
        'resolved': results,
        'errors': errors,
        'stats': resolver.get_cache_stats()
    })

@app.route('/api/resolve/single', methods=['POST'])
def resolve_single_concept():
    """Resolve a single concept"""
    data = request.json
    concept_name = data.get('name', '').strip()
    domain = data.get('domain', 'General')

    if not concept_name:
        return jsonify({'error': 'Concept name required'}), 400

    provider = llm_settings.get_provider()
    if provider is None:
        return jsonify({'error': 'No LLM provider configured'}), 400

    resolver = SemanticResolver(provider)

    try:
        result = resolver.resolve_single(concept_name, domain)
        resolved_concepts[concept_name] = result
        return jsonify({
            'success': True,
            'result': result.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resolve/toggle_discarded', methods=['POST'])
def toggle_discarded():
    """Toggle visibility of discarded concepts"""
    global show_discarded
    show_discarded = not show_discarded

    # Regenerate graph with new visibility setting
    if last_generated_viz is not None and resolved_concepts:
        _regenerate_resolved_graph()

    return jsonify({
        'success': True,
        'show_discarded': show_discarded
    })

@app.route('/api/resolve/clear', methods=['POST'])
def clear_resolutions():
    """Clear all resolved concepts and revert to raw graph"""
    global resolved_concepts
    resolved_concepts.clear()

    # Regenerate original graph
    if last_generated_viz is not None:
        last_generated_viz.save_viz('static/concept_graph.html')

    return jsonify({'success': True})

def _regenerate_resolved_graph():
    """Regenerate the graph visualization with resolved names"""
    if last_generated_viz is None:
        return

    # Create a new network with resolved names
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote', select_menu=True)

    # Apply physics settings
    net.set_options("""
    {
      "nodes": {
        "font": { "size": 16, "face": "tahoma" },
        "borderWidth": 2
      },
      "edges": {
        "color": { "inherit": true },
        "smooth": { "type": "continuous" }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    colors = {"Bio": "#97c2fc", "Tech": "#ffff00", "Social": "#fb7e81", "Physics": "#eb7df4", "Abstract": "#7be141", "Phil": "#ffa500", "General": "#e2e2e2"}

    # Add nodes with resolved names
    for name, concept in last_generated_viz.concepts.items():
        resolution = resolved_concepts.get(name)

        if resolution:
            if resolution.status == "DISCARD" and not show_discarded:
                continue  # Skip discarded concepts if toggle is off

            # Use resolved name as label
            label = resolution.resolved_name if resolution.status == "KEEP" else f"[X] {name}"
            title = f"Original: {name}\nResolved: {resolution.resolved_name}\nDefinition: {resolution.definition}\nConfidence: {resolution.confidence:.0%}"

            if resolution.status == "DISCARD":
                color = "#666666"  # Grey for discarded
                size = 10
            else:
                color = colors.get(concept.domain, "#c2f5ff")
                size = 15 + (concept.dimension * 5)
        else:
            # Not yet resolved - show original
            label = name
            title = f"Props: {list(concept.properties)}\n(Not yet resolved)"
            color = colors.get(concept.domain, "#c2f5ff")
            size = 15 + (concept.dimension * 5)

        net.add_node(name, label=label, title=title, color=color, size=size)

    # Add edges (only for visible nodes)
    visible_nodes = set(net.get_nodes())
    for edge in last_generated_viz.graph.edges(data=True):
        if edge[0] in visible_nodes and edge[1] in visible_nodes:
            net.add_edge(edge[0], edge[1], title=edge[2].get('title', ''))

    net.save_graph('static/concept_graph.html')

# ==========================================
# Save/Load Concept Maps
# ==========================================
@app.route('/api/maps/save', methods=['POST'])
def save_concept_map():
    """Save the current concept map to a JSON file"""
    global last_generated_viz, resolved_concepts

    if last_generated_viz is None:
        return jsonify({'error': 'No concept map to save. Generate one first.'}), 400

    data = request.json
    filename = data.get('filename', '').strip()

    if not filename:
        # Generate a default filename with timestamp
        from datetime import datetime
        filename = f"concept_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Sanitize filename
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    if not filename:
        filename = "concept_map"

    # Build save data
    save_data = {
        "version": "1.0",
        "seeds": [s.to_dict() for s in seeds],
        "domains": domains,
        "concepts": {name: c.to_dict() for name, c in last_generated_viz.concepts.items()},
        "edges": [
            {"from": e[0], "to": e[1], "operation": e[2].get('title', '')}
            for e in last_generated_viz.graph.edges(data=True)
        ],
        "resolutions": {
            name: r.to_dict() for name, r in resolved_concepts.items()
        },
        "settings": {
            "show_discarded": show_discarded
        }
    }

    # Ensure saves directory exists
    saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
    os.makedirs(saves_dir, exist_ok=True)

    filepath = os.path.join(saves_dir, f"{filename}.json")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)

    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': filepath,
        'concept_count': len(last_generated_viz.concepts),
        'resolution_count': len(resolved_concepts)
    })

@app.route('/api/maps/load', methods=['POST'])
def load_concept_map():
    """Load a concept map from a JSON file"""
    global last_generated_viz, resolved_concepts, seeds, domains, show_discarded

    data = request.json
    filename = data.get('filename', '').strip()

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    # Sanitize and find file
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    if not filename.endswith('.json'):
        filename += '.json'

    saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
    filepath = os.path.join(saves_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        # Restore seeds
        seeds.clear()
        for seed_data in save_data.get('seeds', []):
            seeds.append(Concept(
                name=seed_data['name'],
                domain=seed_data['domain'],
                properties=set(seed_data.get('properties', [])),
                dimension=seed_data.get('dimension', 1)
            ))

        # Restore domains
        domains.clear()
        domains.extend(save_data.get('domains', ["Auto-Detect", "Social", "Tech", "Bio", "Physics", "Phil", "Abstract", "General"]))

        # Restore concepts and graph
        viz = VisualGEE()
        for name, concept_data in save_data.get('concepts', {}).items():
            concept = Concept(
                name=concept_data['name'],
                domain=concept_data['domain'],
                properties=set(concept_data.get('properties', [])),
                dimension=concept_data.get('dimension', 1)
            )
            viz.add_concept(concept)

        # Restore edges
        for edge in save_data.get('edges', []):
            viz.graph.add_edge(edge['from'], edge['to'], title=edge.get('operation', ''))

        last_generated_viz = viz

        # Restore resolutions
        resolved_concepts.clear()
        for name, res_data in save_data.get('resolutions', {}).items():
            resolved_concepts[name] = ResolvedConcept(
                original=res_data['original'],
                resolved_name=res_data['resolved_name'],
                definition=res_data.get('definition', ''),
                status=res_data['status'],
                confidence=res_data.get('confidence', 0.5),
                error=res_data.get('error')
            )

        # Restore settings
        settings = save_data.get('settings', {})
        show_discarded = settings.get('show_discarded', True)

        # Regenerate the visualization
        if resolved_concepts:
            _regenerate_resolved_graph()
        else:
            viz.save_viz('static/concept_graph.html')

        return jsonify({
            'success': True,
            'filename': filename,
            'concept_count': len(viz.concepts),
            'resolution_count': len(resolved_concepts),
            'seed_count': len(seeds)
        })

    except Exception as e:
        return jsonify({'error': f'Failed to load: {str(e)}'}), 500

@app.route('/api/maps/list', methods=['GET'])
def list_concept_maps():
    """List all saved concept maps"""
    saves_dir = os.path.join(os.path.dirname(__file__), 'saves')

    if not os.path.exists(saves_dir):
        return jsonify({'maps': []})

    maps = []
    for filename in os.listdir(saves_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(saves_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                maps.append({
                    'filename': filename[:-5],  # Remove .json
                    'concept_count': len(data.get('concepts', {})),
                    'resolution_count': len(data.get('resolutions', {})),
                    'seed_count': len(data.get('seeds', []))
                })
            except Exception:
                maps.append({'filename': filename[:-5], 'error': 'Could not read'})

    return jsonify({'maps': maps})

@app.route('/api/maps/delete', methods=['POST'])
def delete_concept_map():
    """Delete a saved concept map"""
    data = request.json
    filename = data.get('filename', '').strip()

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    if not filename.endswith('.json'):
        filename += '.json'

    saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
    filepath = os.path.join(saves_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filename}'}), 404

    os.remove(filepath)
    return jsonify({'success': True, 'deleted': filename})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5059, help='Port to run the server on')
    args = parser.parse_args()

    os.makedirs('static', exist_ok=True)
    print(f"\n🧠 Concept Explorer starting on http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=args.port)
