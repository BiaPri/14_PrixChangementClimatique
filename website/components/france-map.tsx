"use client";

import { Map } from "@vis.gl/react-maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
// import { StyleSpecification } from "maplibre-gl";

// const customStyle: StyleSpecification = {
//   version: 8,
//   sources: {},
//   layers: [
//     {
//       id: "background",
//       type: "background",
//       paint: {
//         "background-color": "#f0f0f0",
//       },
//     },
//   ],
// };

export function FranceMap() {
  return (
    <Map
      initialViewState={{
        longitude: 2.2137,
        latitude: 46.2276,
        zoom: 5,
      }}
      style={{ width: "100%", height: 400 }}
      mapStyle="https://api.protomaps.com/styles/v5/light/fr.json?key=72196f954acb1cae"
      // mapStyle={customStyle}
    />
  );
}
