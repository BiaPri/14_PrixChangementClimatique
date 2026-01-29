# Data analyse / exploration des données

## Prérequis

Avant tout il est important d'installer `uv` et de `uv sync` pour avoir les packages python.
Se référer au [README.md global](../../README.md) pour cela.

## Téléchargement des données à jour

Tout d'abord 2 types de données sont à disposition :

- les données du projet, créées par l'équipe de Data Engineering (depuis leur dossier /data/dbt_pipeline). Elles auront le format d'une base de donnée : dev.duckdb

- les données odis, des données socio-demo provenant d'un projet DataForGood de la saison précédente. Disponibles pour exploration / utilisation, dans la base de donnée odis.duckdb

Pour télécharger ces deux bases de données, lancer le script suivant (après vous être placé en ligne de commande dans le dossier exploration : `cd data/exploration`) :

```bash
sh download.sh
```

Enfin, il ne faut pas hésiter à chercher, télécharger et utiliser d'autres données sur le web.

Pareil ne pas hésiter à demander conseil sur le Mattermost pour trouver de telles données, ou à l'équipe Data Engineer du projet pour qu'elle ajoute des données dans la base dev.

## (optionnel) Visualisation des données

Avant de se lancer dans la manipulation, il peut être utile d'explorer les données disponibles dans les deux bases de donnée.

L'ui de duckdb est pratique pour cela, lancer (toujours dans le dossier du projet) :

```bash
uv run duckdb dev.duckdb -cmd "ATTACH 'odis.duckdb' AS odis; CALL start_ui();"
```

## Manipulation des données

On peut utiliser n'importe quel outil déjà connu.

3 notebooks d'exemple sont à disposition pour se plonger directement dans les données.

L'ui de duckdb (voir la section précédente de visualisation) propose également des notebooks (attention à bien vérifier comment ils se sauvegardent avant d'aller trop loin).
