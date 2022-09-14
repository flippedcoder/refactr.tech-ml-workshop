import { useState } from 'react'
import { WidgetLoader, Widget } from 'react-cloudinary-upload-widget'
import * as axios from 'axios'

function Classify() {
    const [classification, setClassification] = useState('')
    const [imageUrl, setImageUrl] = useState()

    function getImageLabel(results) {
        console.log('in the classifier')
        setImageUrl(results.info.url)
        console.log(results.info.url)

        const data = {
            image_url: results.info.url
        }

        axios.post("http://127.0.0.1:5000/classify", data).then(res => {
            setClassification(res.data)
        })
    }

    return (
        <div style={{ margin: "24px" }}>
            <WidgetLoader />
            <Widget
                sources={['local', 'camera']}
                cloudName="milecia"
                uploadPreset="tnqzg8z9"
                buttonText="Upload Image"
                onSuccess={res => getImageLabel(res)}
            />
            {
                classification !== "" &&
                <>
                    <h2>{classification.predicted_class}</h2>
                    <h3>Confidence: {classification.probability}</h3>
                    <img src={imageUrl} height={250} width={250} alt={classification.predicted_class * 100} />
                </>
            }
        </div>
    )
}

export default Classify