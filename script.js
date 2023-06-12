const MODEL_PATH = './converted-model/model.json'

async function loadModel(modelPath) {
	return tf.loadLayersModel(modelPath)
}

function preprocessImage(imageData) {
	const input = tf.browser
		.fromPixels(imageData)
		.mean(2)
		.toFloat()
		.div(tf.scalar(255))
	return input.reshape([1, 28, 28, 1])
}

async function predict(model, input) {
	return model.predict(input)
}

async function classifyImage(model, file) {
	try {
		const img = await loadImage(file)

		const imageContainer = document.getElementById('imageContainer')
		// Clear previous image
		imageContainer.innerHTML = ''

		// Display uploaded image
		const image = document.createElement('img')
		image.src = URL.createObjectURL(file)
		imageContainer.appendChild(image)

		const canvas = createCanvas(28, 28)
		const context = canvas.getContext('2d')
		context.drawImage(img, 0, 0, 28, 28)
		const imageData = context.getImageData(0, 0, 28, 28)
		const input = preprocessImage(imageData)

		return predict(model, input)
	} catch (error) {
		console.error('Error occurred while processing the image:', error)
	}
}

function loadImage(file) {
	return new Promise((resolve, reject) => {
		const img = document.createElement('img')
		img.onload = () => resolve(img)
		img.onerror = reject

		// Check if the file is an image
		const reader = new FileReader()
		reader.onload = function (event) {
			img.src = event.target.result
		}
		reader.onerror = reject
		reader.readAsDataURL(file)
	})
}

function createCanvas(width, height) {
	const canvas = document.createElement('canvas')
	canvas.width = width
	canvas.height = height
	return canvas
}

function getTopPrediction(predictions) {
	const highestIndex = predictions.argMax(1).dataSync()[0]
	const highestProbability = predictions.dataSync()[highestIndex]
	return { index: highestIndex, probability: highestProbability }
}

async function handleImageUpload(event) {
	const file = event.target.files[0]
	if (file && file.type.startsWith('image/')) {
		const modelPath = MODEL_PATH
		const model = await loadModel(modelPath)

		// Display the uploaded image immediately
		await classifyImage(model, file)
	}
}

document
	.getElementById('imageUpload')
	.addEventListener('change', handleImageUpload)

document
	.getElementById('classifyButton')
	.addEventListener('click', async () => {
		const modelPath = MODEL_PATH
		const fileInput = document.getElementById('imageUpload')
		const file = fileInput.files[0]

		if (file && file.type.startsWith('image/')) {
			const model = await loadModel(modelPath)
			const predictions = await classifyImage(model, file)
			const result = getTopPrediction(predictions)

			const resultContainer = document.getElementById('result')
			resultContainer.innerHTML = `Class: ${result.index}, Probability: ${result.probability}`
		}
	})
