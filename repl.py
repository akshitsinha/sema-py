import time
from pathlib import Path
from typing import List
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.box import SIMPLE
from text import SemanticSearchEngine
from image import ImageSearchEngine


class SearchREPL:
    def __init__(
        self, directory: str, extensions: List[str], chunk_size: int, chunk_overlap: int
    ):
        self.directory = directory
        self.extensions = extensions
        self.console = Console()

        self.engine = SemanticSearchEngine(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        self.image_engine = ImageSearchEngine()

        self.session = PromptSession(history=FileHistory(".search_history"))

    def run(self):
        self._index()
        self._index_images()
        self._show_welcome()

        while True:
            try:
                user_input = self.session.prompt("> ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                else:
                    self._handle_search(user_input)

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        self.console.print("\n[dim]goodbye[/dim]\n")

    def _index(self, force: bool = False):
        import time

        start_time = time.time()

        files = self.engine._find_files(self.directory, self.extensions)

        if not files:
            self.console.print("[dim]no files found[/dim]\n")
            return

        self.console.print()

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("", total=len(files))

            new_count = 0
            updated_count = 0
            skipped_count = 0
            total_chunks = 0

            for file_path in files:
                if force:
                    if self.engine._get_file_metadata(file_path):
                        self.engine._delete_file_chunks(file_path)
                        updated_count += 1
                    else:
                        new_count += 1

                    chunks = self.engine._index_single_file(file_path)
                    total_chunks += chunks
                else:
                    needs_reindex = self.engine._needs_reindex(file_path)

                    if not self.engine._get_file_metadata(file_path):
                        new_count += 1
                        chunks = self.engine._index_single_file(file_path)
                        total_chunks += chunks
                    elif needs_reindex:
                        updated_count += 1
                        self.engine._delete_file_chunks(file_path)
                        chunks = self.engine._index_single_file(file_path)
                        total_chunks += chunks
                    else:
                        skipped_count += 1

                progress.advance(task)

        elapsed = time.time() - start_time

        parts = []
        if new_count > 0:
            parts.append(f"[cyan]{new_count}[/cyan] new")
        if updated_count > 0:
            parts.append(f"[yellow]{updated_count}[/yellow] updated")
        if skipped_count > 0:
            parts.append(f"[dim]{skipped_count} skipped[/dim]")

        if parts:
            self.console.print(f"[dim]{' · '.join(parts)}[/dim]")

        if total_chunks > 0:
            self.console.print(f"[dim]{total_chunks} chunks indexed[/dim]")

        self.console.print(f"[dim]completed in {elapsed:.2f}s[/dim]")
        self.console.print()

    def _index_images(self, force: bool = False):
        import time

        start_time = time.time()

        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        files = self.image_engine._find_files(self.directory, image_extensions)

        if not files:
            return

        self.console.print()

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("", total=len(files))

            new_count = 0
            updated_count = 0
            skipped_count = 0

            for file_path in files:
                if force:
                    if self.image_engine._get_file_metadata(file_path):
                        self.image_engine._delete_file_chunks(file_path)
                        updated_count += 1
                    else:
                        new_count += 1
                    self.image_engine._index_single_file(file_path)
                else:
                    if not self.image_engine._get_file_metadata(file_path):
                        new_count += 1
                        self.image_engine._index_single_file(file_path)
                    elif self.image_engine._needs_reindex(file_path):
                        updated_count += 1
                        self.image_engine._delete_file_chunks(file_path)
                        self.image_engine._index_single_file(file_path)
                    else:
                        skipped_count += 1

                progress.advance(task)

        elapsed = time.time() - start_time

        parts = []
        if new_count > 0:
            parts.append(f"[cyan]{new_count}[/cyan] new images")
        if updated_count > 0:
            parts.append(f"[yellow]{updated_count}[/yellow] updated images")
        if skipped_count > 0:
            parts.append(f"[dim]{skipped_count} skipped images[/dim]")

        if parts:
            self.console.print(f"[dim]{' · '.join(parts)}[/dim]")

        self.console.print(f"[dim]completed in {elapsed:.2f}s[/dim]")
        self.console.print()

    def _show_welcome(self):
        pass

    def _handle_command(self, command: str) -> bool:
        parts = command.split()
        cmd = parts[0].lower()

        if cmd in ["/exit", "/quit", "/q"]:
            return False

        # Image commands
        if cmd == "/isearch":
            if len(parts) < 2:
                self.console.print("[dim]usage: /isearch <query>[/dim]\n")
            else:
                self._cmd_isearch(" ".join(parts[1:]))
            return True
        elif cmd == "/iref":
            if len(parts) < 2:
                self.console.print("[dim]usage: /iref <filename>[/dim]\n")
            else:
                self._cmd_iref(parts[1])
            return True

        commands = {
            "/help": self._cmd_help,
            "/index": lambda: (self.console.print(), self._index(force=False)),
            "/reindex": lambda: (self.console.print(), self._index(force=True)),
            "/iindex": lambda: (self.console.print(), self._index_images(force=False)),
            "/ireindex": lambda: (self.console.print(), self._index_images(force=True)),
            "/status": self._cmd_status,
            "/istatus": self._cmd_istatus,
            "/config": self._cmd_config,
            "/clear": self._cmd_clear,
            "/iclear": self._cmd_iclear,
            "/files": self._cmd_files,
            "/ifiles": self._cmd_ifiles,
        }

        if cmd in commands:
            commands[cmd]()
        else:
            self.console.print(f"[dim]unknown command:[/dim] {cmd}")
            self.console.print("[dim]type /help for commands[/dim]\n")

        return True

    def _handle_search(self, query: str):
        start = time.time()

        with self.console.status("[dim]searching...", spinner="dots"):
            results = self.engine.search(
                query, n_results=5, directory_filter=self.directory
            )

        elapsed = time.time() - start

        if not results:
            self.console.print("[dim]no results[/dim]\n")
            return

        self.console.print(f"\n[dim]{len(results)} results · {elapsed:.2f}s[/dim]\n")

        for result in results:
            self._display_result(result)

    def _display_result(self, result: dict):
        file_path = result["file_path"]
        score = result["score"]
        line_start = result["line_start"]
        line_end = result["line_end"]

        file_name = Path(file_path).name

        context_text, ctx_start, ctx_end = self.engine.get_context(
            file_path, line_start, line_end, context_lines=5
        )

        lines = context_text.split("\n")
        formatted_lines = []

        for line_num, line in enumerate(lines, start=ctx_start):
            prefix = f"[dim]{line_num:4d} |[/dim] "

            if line_start <= line_num <= line_end:
                formatted_lines.append(prefix + f"[yellow]{line}[/yellow]")
            else:
                formatted_lines.append(prefix + f"[dim]{line}[/dim]")

        content = "\n".join(formatted_lines)

        panel = Panel(
            content,
            title=f"[cyan]{file_name}[/cyan]",
            title_align="left",
            subtitle=f"[dim]lines {line_start}-{line_end} · score {score:.2f}[/dim]",
            subtitle_align="right",
            border_style="bright_black",
            expand=False,
        )

        self.console.print(panel)
        self.console.print()

    def _cmd_help(self):
        commands = [
            ("Text Commands:", ""),
            ("/help", "show commands"),
            ("/index", "reindex text files"),
            ("/reindex", "force reindex text"),
            ("/status", "text stats"),
            ("/config", "show config"),
            ("/files", "list text files"),
            ("/clear", "clear text database"),
            ("", ""),
            ("Image Commands:", ""),
            ("/isearch <query>", "search images by text"),
            ("/iref <filename>", "find similar images"),
            ("/iindex", "reindex images"),
            ("/ireindex", "force reindex images"),
            ("/istatus", "image stats"),
            ("/ifiles", "list images"),
            ("/iclear", "clear image database"),
            ("", ""),
            ("/exit", "exit"),
        ]

        self.console.print()
        for cmd, desc in commands:
            if not cmd:
                self.console.print()
            elif cmd.endswith(":"):
                self.console.print(f"  [bold]{cmd}[/bold]")
            else:
                self.console.print(f"  [cyan]{cmd:<20}[/cyan] [dim]{desc}[/dim]")
        self.console.print()

    def _cmd_status(self):
        stats = self.engine.get_stats()

        table = Table(show_header=True, box=SIMPLE, padding=(0, 2), border_style="cyan")
        table.add_column("metric", style="cyan")
        table.add_column("value", style="dim")

        table.add_row("files", str(stats["total_files"]))
        table.add_row("chunks", str(stats["total_chunks"]))
        table.add_row("size", f"{stats['total_size_mb']} mb")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _cmd_config(self):
        self.console.print()
        self.console.print(f"  [cyan]directory[/cyan]  [dim]{self.directory}[/dim]")
        self.console.print(
            f"  [cyan]extensions[/cyan] [dim]{', '.join(self.extensions)}[/dim]"
        )
        self.console.print(
            f"  [cyan]chunk size[/cyan] [dim]{self.engine.chunk_size}[/dim]"
        )
        self.console.print(
            f"  [cyan]overlap[/cyan]    [dim]{self.engine.chunk_overlap}[/dim]"
        )
        self.console.print(
            f"  [cyan]model[/cyan]      [dim]{self.engine.model_name}[/dim]"
        )
        self.console.print()

    def _cmd_clear(self):
        self.console.print()
        confirm = self.console.input("[dim]clear all data? (yes/no):[/dim] ")

        if confirm.lower() in ["yes", "y"]:
            with self.console.status("[dim]clearing...", spinner="dots"):
                self.engine.clear()
            self.console.print("[dim]cleared[/dim]\n")
        else:
            self.console.print("[dim]cancelled[/dim]\n")

    def _cmd_files(self):
        all_data = self.engine.collection.get()

        if not all_data["ids"] or not all_data["metadatas"]:
            self.console.print("\n[dim]no files indexed[/dim]\n")
            return

        file_chunks = {}
        for metadata in all_data["metadatas"]:
            file_path = metadata["file_path"]
            if file_path not in file_chunks:
                file_chunks[file_path] = 0
            file_chunks[file_path] += 1

        table = Table(show_header=True, box=SIMPLE, padding=(0, 2), border_style="cyan")
        table.add_column("file", style="cyan")
        table.add_column("chunks", style="dim")

        for file_path, chunk_count in sorted(file_chunks.items()):
            file_name = Path(file_path).name
            table.add_row(file_name, str(chunk_count))

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _cmd_isearch(self, query: str):
        start = time.time()

        with self.console.status("[dim]searching images...", spinner="dots"):
            results = self.image_engine.search_by_text(
                query, n_results=5, directory_filter=self.directory
            )

        elapsed = time.time() - start

        if not results:
            self.console.print("\n[dim]no results[/dim]\n")
            return

        self.console.print(f"\n[dim]{len(results)} images · {elapsed:.2f}s[/dim]\n")

        for result in results:
            self._display_image_result(result)

    def _cmd_iref(self, filename: str):
        # Find the full path of the file in the directory
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            potential_path = Path(self.directory) / filename
            if not potential_path.suffix:
                potential_path = Path(self.directory) / f"{filename}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break

        if not image_path:
            self.console.print(f"\n[dim]image not found: {filename}[/dim]\n")
            return

        start = time.time()

        with self.console.status("[dim]finding similar images...", spinner="dots"):
            results = self.image_engine.search_by_image(
                image_path, n_results=5, directory_filter=self.directory
            )

        elapsed = time.time() - start

        if not results:
            self.console.print("\n[dim]no similar images found[/dim]\n")
            return

        self.console.print(
            f"\n[dim]{len(results)} similar images · {elapsed:.2f}s[/dim]\n"
        )

        for result in results:
            self._display_image_result(result)

    def _display_image_result(self, result: dict):
        file_name = result["file_name"]
        score = result["score"]
        names = result.get("extracted_names", "")

        info_parts = [f"score {score:.2f}"]
        if names:
            info_parts.append(f"names: {names}")

        panel = Panel(
            f"[dim]{result['file_path']}[/dim]",
            title=f"[cyan]{file_name}[/cyan]",
            title_align="left",
            subtitle=f"[dim]{' · '.join(info_parts)}[/dim]",
            subtitle_align="right",
            border_style="bright_black",
            expand=False,
        )

        self.console.print(panel)
        self.console.print()

    def _cmd_istatus(self):
        stats = self.image_engine.get_stats()

        table = Table(show_header=True, box=SIMPLE, padding=(0, 2), border_style="cyan")
        table.add_column("metric", style="cyan")
        table.add_column("value", style="dim")

        table.add_row("images", str(stats["total_images"]))
        table.add_row("size", f"{stats['total_size_mb']} mb")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _cmd_iclear(self):
        self.console.print()
        confirm = self.console.input("[dim]clear all image data? (yes/no):[/dim] ")

        if confirm.lower() in ["yes", "y"]:
            with self.console.status("[dim]clearing...", spinner="dots"):
                self.image_engine.clear()
            self.console.print("[dim]cleared[/dim]\n")
        else:
            self.console.print("[dim]cancelled[/dim]\n")

    def _cmd_ifiles(self):
        all_data = self.image_engine.collection.get()

        if not all_data["ids"] or not all_data["metadatas"]:
            self.console.print("\n[dim]no images indexed[/dim]\n")
            return

        table = Table(show_header=True, box=SIMPLE, padding=(0, 2), border_style="cyan")
        table.add_column("file", style="cyan")
        table.add_column("names", style="dim")

        for metadata in all_data["metadatas"]:
            file_name = str(metadata["file_name"])
            names = str(metadata.get("extracted_names", ""))
            table.add_row(file_name, names if names else "[dim]-[/dim]")

        self.console.print()
        self.console.print(table)
        self.console.print()
