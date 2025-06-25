;; Auto-generated. Do not edit!


(when (boundp 'custom_msgs::ChatPrompt)
  (if (not (find-package "CUSTOM_MSGS"))
    (make-package "CUSTOM_MSGS"))
  (shadow 'ChatPrompt (find-package "CUSTOM_MSGS")))
(unless (find-package "CUSTOM_MSGS::CHATPROMPT")
  (make-package "CUSTOM_MSGS::CHATPROMPT"))
(unless (find-package "CUSTOM_MSGS::CHATPROMPTREQUEST")
  (make-package "CUSTOM_MSGS::CHATPROMPTREQUEST"))
(unless (find-package "CUSTOM_MSGS::CHATPROMPTRESPONSE")
  (make-package "CUSTOM_MSGS::CHATPROMPTRESPONSE"))

(in-package "ROS")





(defclass custom_msgs::ChatPromptRequest
  :super ros::object
  :slots (_prompt ))

(defmethod custom_msgs::ChatPromptRequest
  (:init
   (&key
    ((:prompt __prompt) "")
    )
   (send-super :init)
   (setq _prompt (string __prompt))
   self)
  (:prompt
   (&optional __prompt)
   (if __prompt (setq _prompt __prompt)) _prompt)
  (:serialization-length
   ()
   (+
    ;; string _prompt
    4 (length _prompt)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _prompt
       (write-long (length _prompt) s) (princ _prompt s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _prompt
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _prompt (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass custom_msgs::ChatPromptResponse
  :super ros::object
  :slots (_response ))

(defmethod custom_msgs::ChatPromptResponse
  (:init
   (&key
    ((:response __response) "")
    )
   (send-super :init)
   (setq _response (string __response))
   self)
  (:response
   (&optional __response)
   (if __response (setq _response __response)) _response)
  (:serialization-length
   ()
   (+
    ;; string _response
    4 (length _response)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _response
       (write-long (length _response) s) (princ _response s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _response
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _response (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass custom_msgs::ChatPrompt
  :super ros::object
  :slots ())

(setf (get custom_msgs::ChatPrompt :md5sum-) "e0e4e53d8b56a3ee57408d4081b2c754")
(setf (get custom_msgs::ChatPrompt :datatype-) "custom_msgs/ChatPrompt")
(setf (get custom_msgs::ChatPrompt :request) custom_msgs::ChatPromptRequest)
(setf (get custom_msgs::ChatPrompt :response) custom_msgs::ChatPromptResponse)

(defmethod custom_msgs::ChatPromptRequest
  (:response () (instance custom_msgs::ChatPromptResponse :init)))

(setf (get custom_msgs::ChatPromptRequest :md5sum-) "e0e4e53d8b56a3ee57408d4081b2c754")
(setf (get custom_msgs::ChatPromptRequest :datatype-) "custom_msgs/ChatPromptRequest")
(setf (get custom_msgs::ChatPromptRequest :definition-)
      "string prompt
---
string response
")

(setf (get custom_msgs::ChatPromptResponse :md5sum-) "e0e4e53d8b56a3ee57408d4081b2c754")
(setf (get custom_msgs::ChatPromptResponse :datatype-) "custom_msgs/ChatPromptResponse")
(setf (get custom_msgs::ChatPromptResponse :definition-)
      "string prompt
---
string response
")



(provide :custom_msgs/ChatPrompt "e0e4e53d8b56a3ee57408d4081b2c754")


