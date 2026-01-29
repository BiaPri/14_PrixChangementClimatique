#!/bin/sh

# Fonction pour télécharger un fichier
download_file() {
  PUBLIC_URL="$1"
  DESTINATION_PATH="$2"

  echo "Téléchargement de $PUBLIC_URL vers $DESTINATION_PATH..."

  if curl -L "$PUBLIC_URL" -o "$DESTINATION_PATH"; then
    echo "✅ Téléchargement réussi : $DESTINATION_PATH"
  else
    echo "❌ Échec du téléchargement pour $PUBLIC_URL"
  fi
}

# Appel de la fonction pour chaque fichier
download_file "https://s3.fr-par.scw.cloud/qppcc-upload/dev.duckdb" "./dev.duckdb"
download_file "https://s3.fr-par.scw.cloud/qppcc-upload/odis.duckdb" "./odis.duckdb"

echo "Téléchargements terminés !"
