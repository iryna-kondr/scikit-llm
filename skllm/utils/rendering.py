import random
import re
import html
from typing import Dict, List

color_palettes = {
    "light": [
        "lightblue",
        "lightgreen",
        "lightcoral",
        "lightsalmon",
        "lightyellow",
        "lightpink",
        "lightgray",
        "lightcyan",
    ],
    "dark": [
        "darkblue",
        "darkgreen",
        "darkred",
        "darkorange",
        "darkgoldenrod",
        "darkmagenta",
        "darkgray",
        "darkcyan",
    ],
}


def get_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# def validate_text(input_text, output_text):
#     # Verify the original text was not changed (other than addition of tags)
#     stripped_output_text = re.sub(r'<<.*?>>', '', output_text)
#     stripped_output_text = re.sub(r'<</.*?>>', '', stripped_output_text)
#     if not all(word in stripped_output_text.split() for word in input_text.split()):
#         raise ValueError("Original text was altered.")
#     return True


# TODO: In the future this should probably be replaced with a proper HTML template
def render_ner(output_texts, allowed_entities):
    entity_colors = {}
    all_entities = [k.upper() for k in allowed_entities.keys()]

    for i, entity in enumerate(all_entities):
        if i < len(color_palettes["light"]):
            entity_colors[entity] = {
                "light": color_palettes["light"][i],
                "dark": color_palettes["dark"][i],
            }
        else:
            random_color = get_random_color()
            entity_colors[entity] = {"light": random_color, "dark": random_color}

    def replace_match(match):
        reasoning, entity, text = match.groups()
        entity = entity.upper()
        return (
            f'<span class="entity entity-{entity.lower()}" '
            f'title="{entity}: {html.escape(reasoning)}">{text}</span>'
        )

    legend_html = "<div style='margin-bottom: 1em;'>"
    legend_html += "<style>"
    for entity, colors in entity_colors.items():
        description = allowed_entities.get(entity, "No description")
        legend_html += (
            f'.entity-legend-{entity.lower()}-light {{ background-color: {colors["light"]}; color: black; padding: 2px 4px; border-radius: 4px; font-weight: bold; }}\n'
            f'.entity-legend-{entity.lower()}-dark {{ background-color: {colors["dark"]}; color: white; padding: 2px 4px; border-radius: 4px; font-weight: bold; }}\n'
        )
        legend_html += f".entity-legend-{entity.lower()} {{ cursor: pointer; border-radius: 4px; padding: 2px 4px; font-weight: bold; }}\n"
    legend_html += "</style>"
    legend_html += "Entities: "
    for entity in entity_colors.keys():
        description = allowed_entities.get(entity, "No description")
        legend_html += (
            f'<span class="entity-legend-{entity.lower()}" '
            f'title="{entity}: {description}" style="margin-right: 4px;">{entity}</span> '
        )
    legend_html += "</div><hr>"

    css = "<style>:root { --font-size: 16px; }\n"
    css += ".entity { font-size: var(--font-size); padding: 2px 4px; border-radius: 4px; font-weight: bold; }\n"

    light_css = "@media (prefers-color-scheme: light) {\n"
    dark_css = "@media (prefers-color-scheme: dark) {\n"

    for entity, colors in entity_colors.items():
        light_css += f".entity-{entity.lower()} {{ background-color: {colors['light']}; color: black; border-radius: 4px; padding: 2px 4px; font-weight: bold; }}\n"
        dark_css += f".entity-{entity.lower()} {{ background-color: {colors['dark']}; color: white; border-radius: 4px; padding: 2px 4px; font-weight: bold; }}\n"
        light_css += f".entity-legend-{entity.lower()} {{ background-color: {colors['light']}; color: black; border-radius: 4px; padding: 2px 4px; font-weight: bold; }}\n"
        dark_css += f".entity-legend-{entity.lower()} {{ background-color: {colors['dark']}; color: white; border-radius: 4px; padding: 2px 4px; font-weight: bold; }}\n"

    light_css += "}\n"
    dark_css += "}\n"
    css += light_css + dark_css + "</style>"

    rendered_html = ""
    for output_text in output_texts:
        none_pattern = re.compile(r"<not_entity>(.*?)</not_entity>")
        output_text = none_pattern.sub(r'\1', output_text)
        pattern = re.compile(r"<entity><reasoning>(.*?)</reasoning><tag>(.*?)</tag><value>(.*?)</value></entity>")
        highlighted_html = pattern.sub(replace_match, output_text)
        rendered_html += highlighted_html + "<hr>"

    return css + legend_html + rendered_html


def display_ner(output_texts: List[str], allowed_entities: Dict[str, str]):
    rendered_html = render_ner(output_texts, allowed_entities)
    if is_running_in_jupyter():
        from IPython.display import display, HTML 

        display(HTML(rendered_html))
    else:
        with open("skllm_ner_output.html", "w") as f:
            f.write(rendered_html)
        try:
            import webbrowser

            webbrowser.open("skllm_ner_output.html")
        except Exception:
            print(
                "Output saved to 'skllm_ner_output.html', please open it in a browser."
            )


def is_running_in_jupyter():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        return False
    return False
