# Pipeline DBT du projet PCC

## Qu'est ce que c'est ?

[DBT](https://docs.getdbt.com/) est un orchestrateur SQL, c'est à dire qu'il permet d'éffectuer des requêtes SQL dans un order précis. Ces requêtes SQL peuvent dépendre des résultats des précédentes requêtes, ce qui permet in fine une longue transformation des données initiales, au fil des requêtes, pour obtenir des tables dans le format voulu et avec les croisement de données nécessaires.

## Comment la faire tourner ?

Pour faire tourner le dbt de bout en bout, suivre les étapes suivantes :

_(prérequis) Installer les dépendances du projet,_ en installant uv et en faisant `uv sync`

-> voir le README du projet pour plus de détails

_(optionnel) Si vous aviez déjà fait tourner le projet, supprimer l'ancienne base de donnée :_

`rm data/dbt_pipeline/dev.duckdb`

_Télécharger tous les fichiers sources depuis le s3 :_

`uv run python data/utils/download_pipeline_inputs.py`

_Se placer dans le dossier du projet dbt pour le faire tourner :_

`cd data/dbt_pipeline`

_Lancer le seed :_

`uv run dbt seed`

_Lancer le dbt :_

`uv run dbt run`

_Observer le résultat :_

`duckdb --ui dev.duckdb`

## Comment ajouter des données ?

## Plus de doc svp ?

### Tests

- un run dbt test

### Open the doc

- uv run dbt docs generate
- uv run dbt docs serve

### Resources
- Learn more about dbt [in the docs](https://docs.getdbt.com/docs/introduction)
- Check out [Discourse](https://discourse.getdbt.com/) for commonly asked questions and answers
- Join the [chat](https://community.getdbt.com/) on Slack for live discussions and support
- Find [dbt events](https://events.getdbt.com) near you
- Check out [the blog](https://blog.getdbt.com/) for the latest news on dbt's development and best practices
