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


class SearchREPL:
    def __init__(self, directory: str, extensions: List[str], chunk_size: int, chunk_overlap: int):
        self.directory = directory
        self.extensions = extensions
        self.console = Console()
        
        self.engine = SemanticSearchEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.session = PromptSession(
            history=FileHistory('.search_history')
        )
    
    def run(self):
        self._index()
        self._show_welcome()
        
        while True:
            try:
                user_input = self.session.prompt('> ').strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
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
            transient=True
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
        
        self.console.print()
    
    def _show_welcome(self):
        self.console.print()
    
    def _handle_command(self, command: str) -> bool:
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd in ['/exit', '/quit', '/q']:
            return False
        elif cmd == '/help':
            self._cmd_help()
        elif cmd in ['/index', '/reindex']:
            force = cmd == '/reindex'
            self.console.print()
            self._index(force=force)
        elif cmd == '/status':
            self._cmd_status()
        elif cmd == '/config':
            self._cmd_config()
        elif cmd == '/clear':
            self._cmd_clear()
        elif cmd == '/files':
            self._cmd_files()
        else:
            self.console.print(f"[dim]unknown command:[/dim] {cmd}")
            self.console.print("[dim]type /help for commands[/dim]\n")
        
        return True
    
    def _handle_search(self, query: str):
        start = time.time()
        
        with self.console.status("[dim]searching...", spinner="dots"):
            results = self.engine.search(query, n_results=5)
        
        elapsed = time.time() - start
        
        if not results:
            self.console.print("[dim]no results[/dim]\n")
            return
        
        self.console.print(f"\n[dim]{len(results)} results · {elapsed:.2f}s[/dim]\n")
        
        for result in results:
            self._display_result(result)
    
    def _display_result(self, result: dict):
        file_path = result['file_path']
        score = result['score']
        line_start = result['line_start']
        line_end = result['line_end']
        
        file_name = Path(file_path).name
        
        context_text, ctx_start, ctx_end = self.engine.get_context(
            file_path, line_start, line_end, context_lines=5
        )
        
        lines = context_text.split('\n')
        formatted_lines = []
        
        for line_num, line in enumerate(lines, start=ctx_start):
            prefix = f"[dim]{line_num:4d} |[/dim] "
            
            if line_start <= line_num <= line_end:
                formatted_lines.append(prefix + f"[yellow]{line}[/yellow]")
            else:
                formatted_lines.append(prefix + f"[dim]{line}[/dim]")
        
        content = '\n'.join(formatted_lines)
        
        panel = Panel(
            content,
            title=f"[cyan]{file_name}[/cyan]",
            title_align="left",
            subtitle=f"[dim]lines {line_start}-{line_end} · score {score:.2f}[/dim]",
            subtitle_align="right",
            border_style="bright_black",
            expand=False
        )
        
        self.console.print(panel)
        self.console.print()
    
    def _cmd_help(self):
        commands = [
            ('/help', 'show commands'),
            ('/index', 'reindex directory'),
            ('/reindex', 'force reindex'),
            ('/status', 'show stats'),
            ('/config', 'show config'),
            ('/files', 'list files'),
            ('/clear', 'clear database'),
            ('/exit', 'exit'),
        ]
        
        self.console.print()
        for cmd, desc in commands:
            self.console.print(f"  [cyan]{cmd:<12}[/cyan] [dim]{desc}[/dim]")
        self.console.print()
    
    def _cmd_status(self):
        stats = self.engine.get_stats()
        
        table = Table(show_header=True, box=SIMPLE, padding=(0, 2), border_style="cyan")
        table.add_column("metric", style="cyan")
        table.add_column("value", style="dim")
        
        table.add_row("files", str(stats['total_files']))
        table.add_row("chunks", str(stats['total_chunks']))
        table.add_row("size", f"{stats['total_size_mb']} mb")
        
        self.console.print()
        self.console.print(table)
        self.console.print()
    
    def _cmd_config(self):
        self.console.print()
        self.console.print(f"  [cyan]directory[/cyan]  [dim]{self.directory}[/dim]")
        self.console.print(f"  [cyan]extensions[/cyan] [dim]{', '.join(self.extensions)}[/dim]")
        self.console.print(f"  [cyan]chunk size[/cyan] [dim]{self.engine.chunk_size}[/dim]")
        self.console.print(f"  [cyan]overlap[/cyan]    [dim]{self.engine.chunk_overlap}[/dim]")
        self.console.print(f"  [cyan]model[/cyan]      [dim]{self.engine.model_name}[/dim]")
        self.console.print()
    
    def _cmd_clear(self):
        self.console.print()
        confirm = self.console.input("[dim]clear all data? (yes/no):[/dim] ")
        
        if confirm.lower() in ['yes', 'y']:
            with self.console.status("[dim]clearing...", spinner="dots"):
                self.engine.clear()
            self.console.print("[dim]cleared[/dim]\n")
        else:
            self.console.print("[dim]cancelled[/dim]\n")
    
    def _cmd_files(self):
        all_data = self.engine.collection.get()
        
        if not all_data['ids'] or not all_data['metadatas']:
            self.console.print("\n[dim]no files indexed[/dim]\n")
            return
        
        file_chunks = {}
        for metadata in all_data['metadatas']:
            file_path = metadata['file_path']
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
