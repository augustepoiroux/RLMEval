import os
import tempfile

from plasTeX.Compile import parse
from plasTeX.Config import defaultConfig
from plasTeX.DOM import Node

from rlm_eval.utils import ROOT_DIR, logger


def extract_label(node):
    if node.nodeName == "label":
        return node.attributes["label"]
    if node.hasChildNodes:
        for subnode in node.childNodes:
            label = extract_label(subnode)
            if label:
                return label


def extract_text(node):
    text = (
        "".join(
            [
                subsubnode.source
                for subnode in node.childNodes
                for subsubnode in subnode.childNodes
                if subsubnode.nodeName not in ["label", "leanok", "uses", "lean"]
            ]
        )
    ).strip()
    return text.strip()


def extract_uses(node):
    res = {}
    # Extract labels in "uses"
    if "uses" in node.userdata:
        res["uses"] = [extract_label(subnode) for subnode in node.userdata["uses"]]
    return res


def rec_extract_dep_graph_info(node: Node) -> list[dict]:  # type: ignore
    if node.nodeName == "thmenv":
        attributes: dict = {
            "stmt_type": node.thmName,  # type: ignore
            "label": extract_label(node),
            "processed_text": extract_text(node),
            "raw_text": node.source,  # type: ignore
        }

        if "title" in node.attributes and node.attributes["title"]:  # type: ignore
            attributes["title"] = node.attributes["title"].source  # type: ignore

        if node.userdata:
            attributes |= node.userdata
            attributes.pop("lean_urls", None)
            attributes.update(extract_uses(node))

            # Extract proof
            if "proved_by" in attributes:
                attributes["proof"] = {
                    "text": extract_text(attributes["proved_by"]),
                    "source": attributes["proved_by"].source,
                } | attributes["proved_by"].userdata
                attributes["proof"].pop("proves")
                attributes["proof"].update(extract_uses(attributes["proved_by"]))
                attributes.pop("proved_by")

        return [attributes]

    if node.hasChildNodes:
        res = []
        for subnode in node.childNodes:
            res.extend(rec_extract_dep_graph_info(subnode))
        return res


def find_file(root: str, filename: str) -> str | None:
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)


def extract_blueprint_info(blueprint_src_path: str, verbose: bool = False) -> list[dict]:
    # find the webtex file in the blueprint source path
    webtex_file = find_file(blueprint_src_path, "web.tex")
    if not webtex_file:
        raise FileNotFoundError("web.tex file not found in the blueprint source path")

    logger.info(f"Extracting blueprint information from {webtex_file}")

    os.chdir(os.path.dirname(webtex_file))

    config = defaultConfig()
    plastex_file = find_file(blueprint_src_path, "plastex.cfg")
    if plastex_file:
        config.read(plastex_file)
    else:
        logger.warning("No plastex.cfg file found in the blueprint source path")

    if not verbose:
        config["files"]["log"] = True

    tex = parse(webtex_file, config=config)
    doc = tex.ownerDocument

    blueprint_extracted = rec_extract_dep_graph_info(doc)

    return blueprint_extracted


def extract_blueprint_info_content(content: str) -> list[dict]:
    # we copy the blueprint template folder to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        template_dir = os.path.join(ROOT_DIR, "templates", "blueprint")
        os.system(f"cp -r {template_dir} {tmpdirname}")

        # we replace the `content.tex` file with the provided content
        with open(os.path.join(tmpdirname, "blueprint", "src", "content.tex"), "w") as f:
            f.write(content)

        # we extract the blueprint information
        blueprint_extracted = extract_blueprint_info(os.path.join(tmpdirname, "blueprint", "src"))

    return blueprint_extracted
