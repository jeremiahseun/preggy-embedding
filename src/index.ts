import processArticleForEmbedding from "./docs_processing.js";
import { embedDocuments } from "./embed.js";
import { getCollection } from "./qdrant.js";
import { antenatalCareArticle, deliveryArticle, foodsToAvoidArticle, healthyDietArticle, maternityLeaveArticle, postpartumDepressionArticle, prenatalCareArticle, travellingWhilePregnantArticle, twinPregnancyArticle, ultrasoundScansArticle, vegetarianVeganDietArticle, vitaminsSupplementsArticle } from "./utils/articles.js";



async function main() {
    // FIRSTLY GET THE COLLECTION
    await getCollection();
    // PROCESS DOCUMENT AND SEND TO EMBED

   for (const article of articles) {
    console.log("Starting article: ", article.title);
       const processedArticle = processArticleForEmbedding(article)

       if (processedArticle.chunks) {
           // EMBED DOCUMENT (GEMINI LAYER + QDRANT LAYER)
           await embedDocuments(processedArticle);
       }
   }

}

const articles = [
    prenatalCareArticle,
    deliveryArticle,
    twinPregnancyArticle,
    travellingWhilePregnantArticle,
    maternityLeaveArticle,
    postpartumDepressionArticle,
    antenatalCareArticle,
    ultrasoundScansArticle,
    healthyDietArticle,
    vegetarianVeganDietArticle,
    foodsToAvoidArticle,
    vitaminsSupplementsArticle
]

main().catch(console.error);
