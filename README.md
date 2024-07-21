# Sistema de Recomendación de Cursos - Rutas de aprendizaje dinámicas

## Descripción del proyecto

En este proyecto hemos desarrollado un sistema de recomendación de cursos utilizando técnicas avanzadas de procesamiento del lenguaje natural y machine learning. Nuestro objetivo es proporcionar rutas de aprendizaje  personalizadas a los usuarios basadas en su rendimiento en cursos anteriores, de forma que la ruta sea dinámica y adaptada al perfil del alumnado.

## Problema a resolver

El problema que buscamos resolver es la falta de personalización en la recomendación de cursos. En un entorno educativo, es crucial que los estudiantes reciban recomendaciones que se ajusten a su nivel de habilidad y progreso. Nuestro sistema tiene como objetivo mejorar la experiencia de aprendizaje al sugerir el siguiente curso más adecuado.

## Datos utilizados

Para nuestro sistema, hemos utilizado un conjunto de datos de cursos de Udemy en el ámbito de negocios. Los datos incluyen información sobre el título del curso, el nivel, la duración y la categoría. Hemos preprocesado y estructurado estos datos para que sean útiles para nuestro modelo.

## Metodología

### Embeddings: NLP

Utilizamos el modelo `paraphrase-MiniLM-L6-v2` de `sentence-transformers` para generar embeddings de los títulos de los cursos. Estos embeddings nos permiten representar cada curso en un espacio vectorial, capturando el significado y la similitud entre los cursos. El uso de embeddings es crucial porque permite a los modelos entender el contexto y la relación semántica entre diferentes cursos, lo que mejora significativamente la precisión de las recomendaciones.

### Clustering: KMeans

Aplicamos el algoritmo KMeans para agrupar los cursos en clusters basados en sus embeddings. El clustering nos ayuda a identificar cursos similares y organizar el espacio de búsqueda. Cada cluster contiene cursos que son semánticamente similares entre sí, lo que nos permite hacer recomendaciones más precisas y relevantes. El clustering también reduce el espacio de búsqueda, lo que mejora la eficiencia del sistema.

### FAISS: Recuperación de información

Utilizamos FAISS (Facebook AI Similarity Search) para realizar una búsqueda rápida y eficiente de cursos similares. FAISS es una biblioteca que permite la búsqueda de vectores de alta dimensionalidad, lo que es ideal para nuestro caso de uso con embeddings de cursos. La combinación de FAISS con nuestro sistema de embeddings y clustering constituye un método RAG (Retrieval-Augmented Generation) simple pero eficaz. FAISS nos permite recuperar rápidamente los cursos más similares a un curso dado, mejorando la eficiencia y la precisión de nuestras recomendaciones.

### GPT-2: Generación

Para generar recomendaciones personalizadas, utilizamos el modelo GPT-2. Este modelo de lenguaje natural de OpenAI nos permite presentar las recomendaciones de manera más natural y comprensible para los usuarios. GPT-2 genera texto basado en el contexto proporcionado, lo que en nuestro caso incluye información sobre el curso completado y el curso recomendado. Esto añade un nivel de personalización y contexto que enriquece la experiencia del usuario.

## Implementación

Hemos desarrollado una aplicación web utilizando Flask que permite a los usuarios ingresar su rendimiento en un curso y recibir una recomendación personalizada. La aplicación toma en cuenta el tiempo que el usuario ha tardado en completar el curso y la puntuación obtenida en el examen para evaluar su desempeño.

### Evaluación del desempeño

Evaluamos el desempeño del usuario con nuestras reglas:
- Si el usuario obtiene una puntuación superior al 85% y completa el curso en el tiempo estimado, se le considera que "entiende bien".
- Si no, se considera que "necesita mejorar".

### Recomendación del siguiente curso

Basándonos en la evaluación del desempeño, nuestro sistema recomienda el siguiente curso más adecuado utilizando FAISS para la recuperación de información y GPT-2 para generar la recomendación. FAISS busca en los clusters de cursos para encontrar aquellos que son más relevantes según el rendimiento del usuario. Luego, GPT-2 genera una recomendación personalizada que explica por qué se ha seleccionado ese curso y cómo ayudará al usuario a avanzar.

### Guardado del progreso de los usuario

También hemos implementado una funcionalidad para guardar el progreso del usuario, incluyendo el ID del usuario, el curso completado, su desempeño y la siguiente recomendación. Esta información se guarda en un archivo CSV, lo que permite llevar un registro continuo del progreso del usuario y ajustar futuras recomendaciones basadas en su historial.

## Áreas de mejora

Reconocemos que hay áreas de mejora en nuestro sistema:
- **Fine-Tuning**: Podríamos mejorar las recomendaciones ajustando finamente el modelo GPT-2 con datos específicos del dominio. Esto permitiría al modelo generar recomendaciones aún más precisas y contextualmente relevantes.
- **Evaluación continua**: Implementar un sistema de feedback donde los usuarios puedan calificar las recomendaciones para mejorar continuamente el modelo. Esto nos permitiría aprender de los usuarios y ajustar las recomendaciones en tiempo real.
- **Incorporación de otros factores**: Considerar más factores en la recomendación, como los intereses del usuario y tendencias actuales. La incorporación de más datos podría mejorar la personalización y relevancia de las recomendaciones.
- **Mejorar la interfaz de usuario**: Mejorar la interfaz de usuario para hacerla más intuitiva y atractiva. Una interfaz mejorada podría aumentar la participación del usuario y la efectividad del sistema de recomendaciones.

## Conclusiones

Este proyecto proporciona una solución práctica y aplicable para la recomendación personalizada de cursos. Al utilizar técnicas modernas de procesamiento del lenguaje natural y machine learning, hemos desarrollado un sistema eficiente y eficaz. Aunque hay margen para mejoras, este trabajo sienta una base sólida para futuras investigaciones y desarrollos en el campo de la personalización educativa.

## Extra

También hemos hecho el archivo app_streamlit.py y streamlit.py para correr la aplicación a través de streamlit como punto extra.

Enlace al vídeo en Linkedin: https://www.linkedin.com/posts/saradiazpmartin_tras-la-charla-edtech-de-esta-semana-y-el-activity-7220763043251576832-tcVC?utm_source=share&utm_medium=member_ios
--

¡Gracias!

Sara Díaz
