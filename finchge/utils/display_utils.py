from typing import Iterable, Union


# Display utils
def highlight_bnf(grammar_text: Union[str, Iterable[str]]) -> str:
    """Grammar highlighter that returns HTML-formatted text."""
    lines: list[str]

    if isinstance(grammar_text, str):
        lines = grammar_text.strip().split("\n")
    else:
        lines = list(grammar_text)

    nonterminal_style = '<b style="color: #d33682;">'
    operator_style = '<span style="color: #666;">::=</span>'
    pipe_style = '<span style="color: #859900;">|</span>'
    rhs_style = '<span style="color: #268bd2;">{}</span>'

    html_lines: list[str] = []

    for line in lines:
        if "::=" not in line:
            html_lines.append(line)
            continue

        lhs, rhs = line.split(" ::= ", 1)

        lhs = lhs.replace("<", f"{nonterminal_style}&lt;").replace(">", "&gt;</b>")

        rhs = (
            rhs.replace("<", f"{nonterminal_style}&lt;")
            .replace(">", "&gt;</b>")
            .replace(" | ", f" {pipe_style} ")
        )

        formatted_line = f"{lhs} {operator_style} {rhs_style.format(rhs)}"
        html_lines.append(formatted_line)

    return "<br>".join(html_lines)


def is_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()  # type: ignore
        return ip is not None and hasattr(ip, "config")
    except ImportError:
        return False


def display_html(html_content: str) -> None:
    """
    Display HTML content
    """
    from IPython.display import HTML, display

    display(HTML(html_content))  # type: ignore[no-untyped-call]


def display_markdown(md_content: str) -> None:
    """
    Display Markdown content.
    """
    if is_jupyter():
        try:
            from IPython.display import Markdown, display

            display(Markdown(md_content))  # type: ignore[no-untyped-call]
        except ImportError:
            print(md_content)
    else:
        print(md_content)
