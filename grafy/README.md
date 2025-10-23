# Nástroj pro práci s `.tg` grafy

Tento repozitář obsahuje jednoduchý Python skript `graph_tool.py`, který umí načítat grafy ze souborů typu `.tg` a poskytuje základní analýzy (shrnutí grafu, nejkratší cesta, minimální kostra, centrality, komponenty, detekce cyklů, topologické řazení, dávkové reporty a další).

Součásti projektu:
- `graph_tool.py` — hlavní CLI skript (interaktivní menu + nepřerušené sub-příkazy)
- `requirements.txt` — doporučená závislost (`networkx`)
- `01.tg` … `20.tg` — testovací grafy (vámi dodané)
- `tg-c01.pdf` — zadání / instrukce (vámi dodané)

Obsah této dokumentace:
- Instalace (včetně offline provozu)
- Jak spouštět (PowerShell příklady)
- Interaktivní menu
- Popis hlavních příkazů
- Formát vstupních `.tg` souborů
- Tipy a často používané scénáře

Poznámka k jazyku: dokumentace a výstupy nástroje jsou v češtině.

## Požadavky a offline použití

- Doporučená závislost: `networkx` (v `requirements.txt`).
- Nástroj ale poběží i bez `networkx` — obsahuje vestavěnou lehkou implementaci grafu (fallback), takže je použitelný na strojích bez přístupu k internetu.

Pokud chcete nainstalovat `networkx` (pokud máte internet nebo offline wheel), spusťte:

```powershell
python -m pip install -r requirements.txt
```

## Rychlý start (PowerShell)

Zpracování jednoho souboru (shrnutí):

```powershell
python graph_tool.py summary 01.tg
```

Nejkratší cesta (z uzlu A do F):

```powershell
python graph_tool.py shortest 01.tg --src A --dst F
```

Centrálnosti (top 5):

```powershell
python graph_tool.py centrality 01.tg --top 5
```

Dávkový report (uloží JSON report všech `*.tg` v adresáři):

```powershell
python graph_tool.py batch --dir . --out report.json
```

Analýza (matice sousednosti, incidence, stupně, adjacency list):

```powershell
python graph_tool.py analyze 01.tg
```

Zobrazení definic pojmů (čeština):

```powershell
python graph_tool.py define graf
python graph_tool.py define "matice sousednosti"
```

## Interaktivní menu

Pokud spustíte program bez sub-příkazu, zobrazí se číslované menu, které umožní vybrat akci zadáním čísla. Po výběru se program zeptá na potřebné parametry (cestu k .tg souboru, názvy uzlů apod.).

Spuštění interaktivního režimu:

```powershell
python graph_tool.py
```

Průběh: vyberete číslo volby, zadáte název souboru (pokud je potřeba) a další parametry.

## Popis hlavních příkazů

- summary <file.tg>
	- tiskne základní charakteristiky grafu (orientovanost, počet uzlů/hrán, zda je vážený, počet komponent, stupně uzlů atd.)
- shortest <file.tg> --src X --dst Y
	- vypočte nejkratší cestu mezi uzly X a Y; pokud graf obsahuje váhy, použije Dijkstra, jinak BFS
- mst <file.tg>
	- pokud je graf neorientovaný (či po převodu na neorientovaný), vytiskne hrany minimální kostry
- centrality <file.tg> [--top K]
	- vypíše nejdůležitější uzly podle stupně a betweenness centrality
- components <file.tg>
	- vypíše (silně) souvislé komponenty
- cycles <file.tg>
	- najde cykly v grafu (pro orientované i neorientované)
- topo <file.tg>
	- topologické řazení (jen pro orientované DAG)
- analyze <file.tg>
	- komplexní analýza: shrnutí, matice sousednosti, matice incidence, stupně uzlů a sousední seznamy
- define <term> [...]
	- vrátí českou definici zadaného termínu (např. `graf`, `uzel`, `matice sousednosti`)
- batch [--dir DIR] [--out FILE]
	- provede `summary` pro všechny `.tg` v adresáři a uloží JSON report

## Formát `.tg` vstupních souborů

Skript očekává textový formát podobný ukázkám v zadání. Podporovány tvary:

- Deklarace uzlu:
	- `u A;` — uzel s identifikátorem A
	- `u A[3];` nebo `u A 3;` — uzel A s volitelným číslem/ohodnocením
	- `u *;` — vynechaný/„vynucený“ uzel (parser ignoruje)

- Hrany (příklady):
	- `h A > B 3 :h1;` — orientovaná hrana A -> B s vahou 3 a označením `h1`
	- `h A < B 2 :h2;` — interpretováno jako B -> A
	- `h A - B 5;` — neorientovaná hrana mezi A a B, váha 5

Pořadí řádků ve vstupním souboru je libovolné (jen s drobnými výjimkami v zadání). Parser je tolerantní k mezerám a volitelným částem (váha/label).

## Příklady použití

Shrnutí grafu `01.tg`:

```powershell
python graph_tool.py summary 01.tg
```

Interaktivní volba: spustíte `python graph_tool.py`, zadáte číslo volby `1` (Summary), poté zadáte `01.tg` atd.

Dávkové vytvoření reportu pro všechny `.tg` v aktuálním adresáři:

```powershell
python graph_tool.py batch --dir . --out report.json
```

Export rychlé analýzy do terminálu (matice a seznamy):

```powershell
python graph_tool.py analyze 01.tg
```

Zobrazení definice pojmu v češtině:

```powershell
python graph_tool.py define graf
python graph_tool.py define "matice sousednosti"
```

## Poznámky k nasazení bez internetu

- Skript má vestavěný fallback (SimpleGraph), takže běží i bez `networkx`. Fallback poskytuje základní algoritmy potřebné pro výuku a testy. Pokud však chcete plně robustní a rychlé řešení (a máte možnost nainstalovat závislosti), nainstalujte `networkx`.
- Pokud potřebujete zcela offline instalaci `networkx`, doporučuji stáhnout wheel (.whl) na stroji s internetem a přenést ho na cílový stroj.

## Další rozšíření (možnosti)

- export do GraphML/GEXF nebo PNG pro vizualizaci
- více statistik (closeness/eigenvector centrality, detekce mostů a artikulačních bodů)
- unit-testy a skripty pro automatické hodnocení na 01.tg–20.tg

Pokud chcete, mohu některé z těchto rozšíření rovnou implementovat. Napište, kterou funkcionalitu upřednostňujete.

