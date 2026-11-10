# Inventario de cierre académico

Fecha de corte: 7 de noviembre de 2026. Este paquete resume el estado verificado para el informe y la sustentación. Power BI no se modificó en este cierre.

## Resultados consolidados

| Área | Resultado comprobado | Alcance |
|---|---|---|
| API v1.2 | `/health` HTTP 200; PostgreSQL conectado; `/schema` con 34 variables y `/schema/v9d` con 7 | Contratos consultados en el servicio local activo |
| PostgreSQL | 9 filas al momento de la consulta | La verificación de este cierre fue de solo lectura y no insertó ni modificó filas |
| V8B | Accuracy 0,9011; balanced accuracy 0,6791; macro-F1 0,6674; precision/recall/F1 de Con Muertos 0,1753/0,2421/0,2034 | Prueba aleatoria estratificada, 39.227 registros; 34 variables, varias posteriores al siniestro |
| V9D | Accuracy 0,5295; balanced accuracy 0,4011; macro-F1 0,3768; precision/recall/F1 de Con Muertos 0,0250/0,1080/0,0405 | Prueba temporal 2019–2020, 57.699 registros; siete variables temporales/localidad |
| Geografía | 50 candidatas: 43 ambiguas, 7 únicas pendientes, 0 aprobadas humanamente | 404 IDs de siniestro coinciden textualmente con dirección y localidad de las 7 únicas pendientes; no son geolocalizaciones validadas |
| Prevención | No viable entrenar o validar con los datos actuales | No hay negativos confirmados ni exposición; V8B/V9D no predicen ocurrencia |

Las métricas de V8B y V9D no son directamente comparables porque difieren las variables y el protocolo de evaluación. Ningún registro de prueba API se incorpora como dato de modelado.

## Archivos académicos

- `informe_academico_final_borrador.docx`: borrador completo, ocho páginas, con objetivos, metodología, arquitectura, resultados, limitaciones, conclusiones, recomendaciones y anexos. Los datos de autoría, institución, programa y dirección están pendientes de completar.
- `presentacion_sustentacion.pptx`: 12 diapositivas editables con notas del presentador y fuentes.
- `inventario_final_y_pendientes.md`: este inventario.

## Evidencia geográfica

- `geografia/revision_50_candidatas_y_coincidencias.csv`: inventario de las 50 filas, estados sin alterar y conteo por coincidencia textual normalizada de dirección/localidad.
- `geografia/7_candidatas_unicas_pendientes.csv`: siete ubicaciones con un candidato geométrico y coincidencias textuales, todas marcadas `VALIDADO=NO`.
- `geografia/7_candidatas_unicas_pendientes.geojson`: capa de puntos candidatos, con la advertencia de que coordenadas y asociación requieren validación humana.
- `geografia/resumen_geografico_academico.json`: método, conteos y límites de interpretación.
- Scripts reproducibles: `src/Dataset4/generar_resultado_geografico_academico.py` y `src/api/verificar_integracion_api_postgresql.py`.

Los 404 IDs se reparten entre las siete combinaciones distintas de dirección/localidad. La coincidencia textual justifica una asociación con la dirección candidata, pero no una coordenada exacta del accidente. Las filas ambiguas pueden repetir una misma clave espacial, por lo que sus conteos no se suman.

## Diagnóstico preventivo incorporado

La carpeta `diagnostico_preventivo/` contiene el README, el resumen JSON y las tablas anual, de split temporal, geográfica y horaria. El panel hipotético contiene 2.630.400 celdas de candidato-fecha-hora: 3.513 con reporte textual y 2.626.887 sin reporte registrado. Estos huecos no son negativos confirmados. El split train 2015–2018, validation 2019 y test 2020 queda definido, pero sin ejecutar. No se reportan PR-AUC, Brier ni recall@K para ocurrencia.

El diagnóstico fuente está en `models/diagnostico_panel_interseccion_hora.json`; el script reproducible está en `src/Dataset4/diagnostico_panel_interseccion_hora.py`.

## Pruebas realizadas

- Suite automatizada: 19 de 19 aprobadas con `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`.
- Salida completa de la suite guardada en `pruebas/resultado_suite_unittest.txt`.
- Verificación en vivo de solo lectura: `/health`, `/schema`, `/schema/v9d` y conexión PostgreSQL. Resultado guardado en `pruebas/resultado_integracion_solo_lectura.json`.
- La consulta identificó ID 8 como registro técnico marcado y ID 9 como escenario V9D sintético con `ANO=2026`, fuera de 2015–2020. Ambos se excluyen de resultados reales. La prueba de cierre no insertó datos.
- Las inserciones reales anteriores ya dejaron los IDs 8 y 9 como pruebas técnicas persistentes. No se alteraron los primeros siete registros.
- El PPTX contiene 12 diapositivas, cinco tablas editables, notas por diapositiva y pasó las comprobaciones de integridad y geometría del paquete. Se inspeccionaron visualmente sus 12 vistas renderizadas.
- El DOCX se exportó desde Microsoft Word a PDF y se revisaron sus ocho páginas. El renderizador automatizado no pudo ejecutarse porque LibreOffice no está instalado; la revisión se completó con la instalación local de Word y Poppler.

## Errores o incidencias

- No quedaron fallos en la API, PostgreSQL, las pruebas automatizadas ni el generador geográfico.
- La primera validación técnica del PPTX no encontró el módulo Node del entorno porque faltaba la variable requerida por el verificador. Se repitió con la ruta de dependencias correcta y pasó; no afectó el archivo final.
- El primer intento de render DOCX falló por ausencia de `soffice.exe`. Se usó Word local y se corrigió una página en blanco del primer borrador; el DOCX final tiene ocho páginas sin página adicional.

## Pendientes que requieren intervención humana

1. Completar autoría, institución, programa, dirección académica y formato/citas requeridos por la institución.
2. Validar manualmente las siete candidatas únicas y resolver las 43 ambiguas. Registrar fuente, persona revisora, fecha, coordenada aprobada y observaciones sin sobrescribir las coordenadas calculadas originales.
3. Revisar las reglas de exclusión de los IDs 8 y 9 en cualquier consulta o tablero. Permanecen en PostgreSQL como registros técnicos y no deben tratarse como siniestros reales.
4. La responsable de Power BI debe integrar su versión final con estos límites: gravedad condicionada a siniestros registrados, geografía candidata y diagnóstico preventivo no viable.
5. Para reabrir un experimento preventivo, conseguir universo espacial validado, evidencia de cobertura del registro y exposición por intersección-hora. Después definir negativos y ejecutar el split temporal acordado.

