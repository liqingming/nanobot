"""Terminal Markdown helpers."""

from __future__ import annotations

from typing import ClassVar

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown, MarkdownElement
from rich.syntax import Syntax


class TransparentCodeBlock(CodeBlock):
    """Rich Markdown code block without a theme-colored background."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme="ansi_dark",
            word_wrap=True,
            background_color="default",
            padding=(0, 1),
        )


class TerminalMarkdown(Markdown):
    """Markdown renderable tuned for chat output in terminals."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "fence": TransparentCodeBlock,
        "code_block": TransparentCodeBlock,
    }


def terminal_markdown(markup: str) -> TerminalMarkdown:
    return TerminalMarkdown(markup)
