import re


def filter_xml_tags(xml_string, tags):
    pattern = "|".join(f"<{tag}>.*?</{tag}>" for tag in tags)
    regex = re.compile(pattern, re.DOTALL)
    matches = regex.findall(xml_string)
    return "".join(matches)


def filter_unwanted_entities(xml_string, allowed_entities):
    allowed_values_pattern = "|".join(allowed_entities.keys())
    replacement = r"<not_entity>\3</not_entity>"
    pattern = rf"<entity><reasoning>(.*?)</reasoning><tag>(?!{allowed_values_pattern})(.*?)</tag><value>(.*?)</value></entity>"
    return re.sub(pattern, replacement, xml_string)


def replace_all_at_once(text, replacements):
    sorted_keys = sorted(replacements, key=len, reverse=True)
    regex = re.compile(r"(" + "|".join(map(re.escape, sorted_keys)) + r")")
    return regex.sub(lambda match: replacements[match.group(0)], text)


def json_to_xml(
    original_text: str,
    tags: list,
    tag_root: str,
    non_tag_root: str,
    value_key: str = "value",
    attributes: list = None,
):

    if len(tags) == 0:
        return f"<{non_tag_root}>{original_text}</{non_tag_root}>"

    if attributes is None:
        attributes = tags[0].keys()

    replacements = {}
    for item in tags:
        value = item.get(value_key, "")
        if not value:
            continue

        attribute_parts = []
        for attr in attributes:
            if attr in item:
                attribute_parts.append(f"<{attr}>{item[attr]}</{attr}>")
        attribute_str = "".join(attribute_parts)
        replacements[value] = f"<{tag_root}>{attribute_str}</{tag_root}>"
    original_text = replace_all_at_once(original_text, replacements)

    parts = re.split(f"(<{tag_root}>.*?</{tag_root}>)", original_text)
    final_text = ""
    for part in parts:
        if not part.startswith(f"<{tag_root}>"):
            final_text += f"<{non_tag_root}>{part}</{non_tag_root}>"
        else:
            final_text += part
    return final_text
